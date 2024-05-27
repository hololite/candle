// An implementation of LLaMA https://github.com/facebookresearch/llama
//
// This is based on nanoGPT in a similar way to:
// https://github.com/Lightning-AI/lit-llama/blob/main/lit_llama/model.py
//
// The tokenizer config can be retrieved from:
// https://huggingface.co/hf-internal-testing/llama-tokenizer/raw/main/tokenizer.json

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use std::fs;
use anyhow::{bail, Error as E, Result};
use clap::{Parser, ValueEnum};
use tokenizers::Tokenizer;
use tracing_chrome::ChromeLayerBuilder;
use tracing_subscriber::prelude::*;
use serde::Deserialize;
use std::io::{self, Write};

use hf_hub::{api::sync::{ApiBuilder, ApiRepo, ApiError}, Repo, RepoType};
use candle::{DType, Tensor, Device};
use candle_nn::VarBuilder;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::llama as model;
use model::{Llama, Cache, LlamaConfig, Config};

const EOS_TOKEN: &str = "</s>";
const POST_PROMPT : &str = ". Answer the question concisely. Do not discuss other irrelevant stuff.";

#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum)]
enum ModelVer {
    V1,
    V2,
    V3,
    V3Instruct,
    #[value(name = "solar-10.7b")]
    Solar10_7B,
    #[value(name = "tiny-llama-1.1b-chat")]
    TinyLlama1_1BChat,
}

#[derive(Clone, Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// The temperature used to generate samples.
    #[arg(long, default_value_t = 0.65)]
    temperature: f64,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_p: Option<f64>,

    /// Only sample among the top K samples.
    #[arg(long)]
    top_k: Option<usize>,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// The length of the sample to generate (in tokens).
    #[arg(short = 'n', long, default_value_t = 10000)]
    sample_len: usize,

    /// Disable the key-value cache.
    #[arg(long)]
    no_kv_cache: bool,

    /// The initial prompt.
    #[arg(long)]
    prompt: Option<String>,

    /// Use different dtype than f16
    #[arg(long)]
    dtype: Option<String>,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    #[arg(long)]
    model_id: Option<String>,

    #[arg(long)]
    revision: Option<String>,

    /// The model size to use.
    #[arg(long, default_value = "v3-instruct")]
    model_ver: ModelVer,

    #[arg(long)]
    use_flash_attn: bool,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 128)]
    repeat_last_n: usize,
}

impl Args {
    fn parse_args() -> Self {
        let args = Args::parse();
        let _guard = if args.tracing {
            let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
            tracing_subscriber::registry().with(chrome_layer).init();
            Some(guard)
        } else {
            None
        };

        println!(
            "avx={}, neon={}, simd128={}, f16c={}",
            candle::utils::with_avx(),
            candle::utils::with_neon(),
            candle::utils::with_simd128(),
            candle::utils::with_f16c()
        );

        println!(
            "model-id={:?}, model-ver={:?}, dtype={:?}, cpu={}, tracing={}",
            args.model_id,
            args.model_ver,
            args.dtype,
            args.cpu,
            args.tracing
        );

        println!(
            "temp={:.2}, top-p={:?}, top-k={:?}, repeat-penalty={:.2}\nrepeat-last-n={}, seed={}, sample-len={}\n",
            args.temperature,
            args.top_p,
            args.top_k,
            args.repeat_penalty,
            args.repeat_last_n,
            args.seed,
            args.sample_len
        );

        args
    }
}

#[derive(Clone)]
struct ModelParams {
    quantized: bool,
    use_flash_attn: bool,
    dtype: Option<String>,
    model_ver: ModelVer,
    model_id: Option<String>,
    revision: Option<String>
}
impl ModelParams {
    fn new(quantized: bool, use_flash_attn: bool, dtype: Option<String>, model_ver: ModelVer, model_id: Option<String>, revision: Option<String>) -> Self {
        Self {
            quantized,
            use_flash_attn,
            dtype,
            model_ver,
            model_id,
            revision
        }
    }
}

#[derive(Deserialize)]
struct AppConfig {
    api_key: String,
}

fn read_config(file_path: &str) -> Result<AppConfig, Box<dyn std::error::Error>> {
    // Read the file content
    let config_content = fs::read_to_string(file_path)?;
    // Parse the JSON content
    let config: AppConfig = serde_json::from_str(&config_content)?;
    Ok(config)
}

fn initialize_api_repo(access_token: &str, model_params: &ModelParams) -> Result<ApiRepo, ApiError> {
    println!("initializing api repo...");

    let api_builder = ApiBuilder::new();
    let api_builder_token =  api_builder.with_token(Some(String::from(access_token)));
    let api = api_builder_token.build()?;

    let model_id =
        if model_params.quantized {
            "lmz/candle-quantized-phi".to_string()
        } else {
            match model_params.model_ver {
                ModelVer::V1 => "Narsil/amall-7b".to_string(),
                ModelVer::V2 => "meta-llama/Llama-2-7b-hf".to_string(),
                ModelVer::V3 => "meta-llama/Meta-Llama-3-8B".to_string(),
                ModelVer::V3Instruct => "meta-llama/Meta-Llama-3-8B-Instruct".to_string(),
                ModelVer::Solar10_7B => "upstage/SOLAR-10.7B-v1.0".to_string(),
                ModelVer::TinyLlama1_1BChat => "TinyLlama/TinyLlama-1.1B-Chat-v1.0".to_string(),
            }
        };

    let revision = &model_params.revision.clone().unwrap_or("main".to_string());
    let repo = api.repo(Repo::with_revision(model_id, RepoType::Model, revision.to_string()));

    Ok(repo)
}
struct ModelInfo {
    llama: Llama,
    tokenizer: Tokenizer,
    cache: Cache,
    config: Config
}


fn load_model(api_repo: &ApiRepo, device: &Device, model_params: &ModelParams) -> Result<ModelInfo, E> {
    let dtype = match model_params.dtype.as_deref() {
        Some("f16") => DType::F16,
        Some("bf16") => DType::BF16,
        Some("f32") => DType::F32,
        Some(dtype) => bail!("Unsupported dtype {dtype}"),
        None => DType::F16,
    };

    //let (llama, tokenizer_filename, mut cache, config) = {
    let model_id = model_params.model_id.clone();
    let model_id = model_id.unwrap_or_else(|| match model_params.model_ver {
        ModelVer::V1 => "Narsil/amall-7b".to_string(),
        ModelVer::V2 => "meta-llama/Llama-2-7b-hf".to_string(),
        ModelVer::V3 => "meta-llama/Meta-Llama-3-8B".to_string(),
        ModelVer::V3Instruct => "meta-llama/Meta-Llama-3-8B-Instruct".to_string(),
        ModelVer::Solar10_7B => "upstage/SOLAR-10.7B-v1.0".to_string(),
        ModelVer::TinyLlama1_1BChat => "TinyLlama/TinyLlama-1.1B-Chat-v1.0".to_string(),
    });
    println!("loading the model weights from {model_id}");

    let tokenizer_filename = api_repo.get("tokenizer.json")?;
    let config_filename = api_repo.get("config.json")?;
    let lconfig: LlamaConfig = serde_json::from_slice(&std::fs::read(config_filename)?)?;
    let config = lconfig.into_config(model_params.use_flash_attn);

    let filenames = match model_params.model_ver {
        ModelVer::V1 | ModelVer::V2 | ModelVer::V3 | ModelVer::V3Instruct | ModelVer::Solar10_7B => {
            candle_examples::hub_load_safetensors(&api_repo, "model.safetensors.index.json")?
        }
        ModelVer::TinyLlama1_1BChat => vec![api_repo.get("model.safetensors")?],
    };
    let no_kv_cache = false;
    let cache = model::Cache::new(!no_kv_cache, dtype, &config, device)?;

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };
    //(Llama::load(vb, &config)?, tokenizer_filename, cache, config)
    let llama = Llama::load(vb, &config)?;
    //};

    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    let model_info = ModelInfo {
        llama,
        tokenizer,
        cache,
        config
    };

    Ok(model_info)
}

struct PromptParams {
    temperature: f64,
    top_k: Option<usize>,
    top_p: Option<f64>,
    seed: u64,
    sample_len: usize,
    repeat_penalty: f32,
    repeat_last_n: usize
}

fn submit_prompt(model_info: &ModelInfo, device: &Device, prompt: &str, args: &PromptParams) -> Result<(), E> {
    let eos_token_id = model_info.config
        .eos_token_id
        .or_else(|| model_info.tokenizer.token_to_id(EOS_TOKEN));

    //let prompt = args.prompt.as_ref().map_or(DEFAULT_PROMPT, |p| p.as_str());
    let mut tokens = model_info.tokenizer
        .encode(prompt, true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();
    let mut tos = candle_examples::token_output_stream::TokenOutputStream::new(model_info.tokenizer.clone());

    println!("starting the inference loop");
    println!("#### prompt=[{prompt}]\n");
    let mut logits_processor = {
        let temperature = args.temperature;
        let sampling = if temperature <= 0. {
            Sampling::ArgMax
        } else {
            match (args.top_k, args.top_p) {
                (None, None) => Sampling::All { temperature },
                (Some(k), None) => Sampling::TopK { k, temperature },
                (None, Some(p)) => Sampling::TopP { p, temperature },
                (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature },
            }
        };
        LogitsProcessor::from_sampling(args.seed, sampling)
    };

    let mut start_gen = std::time::Instant::now();
    let mut index_pos = 0;
    let mut token_generated = 0;

    let mut cache = model_info.cache.clone();
    for index in 0..args.sample_len {
        let (context_size, context_index) = if cache.use_kv_cache && index > 0 {
            (1, index_pos)
        } else {
            (tokens.len(), 0)
        };
        if index == 1 {
            start_gen = std::time::Instant::now()
        }
        let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
        let input = Tensor::new(ctxt, device)?.unsqueeze(0)?;
        let logits = model_info.llama.forward(&input, context_index, &mut cache)?;
        let logits = logits.squeeze(0)?;
        let logits = if args.repeat_penalty == 1. {
            logits
        } else {
            let start_at = tokens.len().saturating_sub(args.repeat_last_n);
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                args.repeat_penalty,
                &tokens[start_at..],
            )?
        };
        index_pos += ctxt.len();

        let next_token = logits_processor.sample(&logits)?;
        token_generated += 1;
        tokens.push(next_token);

        if Some(next_token) == eos_token_id {
            break;
        }
        if let Some(t) = tos.next_token(next_token)? {
            print!("{t}");
            std::io::stdout().flush()?;
        }
    }
    if let Some(rest) = tos.decode_rest().map_err(E::msg)? {
        print!("{rest}");
    }
    let dt = start_gen.elapsed();
    println!(
        "\n\n{} tokens generated ({} token/s)\n",
        token_generated,
        (token_generated - 1) as f64 / dt.as_secs_f64(),
    );

    Ok(())
}

fn main() -> Result<()> {
    let config_file_path = ".config";
    let access_token: String;
    match read_config(config_file_path) {
        Ok(config) => access_token = config.api_key,
        Err(e) => panic!("Error reading config file: {}", e),
    }

    let args = Args::parse_args();

    let device = candle_examples::device(args.cpu)?;

    let model_params = ModelParams::new(
        false,
        args.use_flash_attn,
        args.dtype.clone(),
        args.model_ver,
        args.model_id.clone(),
        args.revision.clone()
    );

    let api_repo = initialize_api_repo(&access_token, &model_params)?;

    let model_info = load_model(&api_repo, &device, &model_params)?;

    let prompt_params  = PromptParams {
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: args.top_p,
        seed: args.seed,
        sample_len: args.sample_len,
        repeat_penalty: args.repeat_penalty,
        repeat_last_n: args.repeat_last_n
    };

    if args.prompt.is_none() {
        loop {
            print!("> ");
            io::stdout().flush().unwrap();
            let mut prompt = String::new();

            // Read the input from stdin
            match io::stdin().read_line(&mut prompt) {
                Ok(0) => {
                    // Control-D is pressed, as no bytes were read
                    println!("exiting...");
                    break;
                }
                Ok(_) => {
                    // Remove the newline character from the end of the input
                    prompt = prompt.trim_end().to_string();
                    if prompt.len() == 0 {
                        continue;
                    }
                    prompt = format!("{} {}", prompt, POST_PROMPT);

                    submit_prompt(&model_info, &device, &prompt, &prompt_params)?;
                }
                Err(error) => {
                    // Handle any errors that occur during reading
                    eprintln!("Error reading input: {}", error);
                    break;
                }
            }
        }
    }
    else {
        let prompt = format!("{} {}", args.prompt.unwrap(), POST_PROMPT);
        submit_prompt(&model_info, &device, &prompt, &prompt_params)?;
    }

    Ok(())
}