#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
//use candle_transformers::models::segment_anything::prompt_encoder;
//use candle_transformers::models::whisper::model;
use clap::{Parser, ValueEnum};
use hf_hub::api::sync::ApiError;
use tracing_chrome::ChromeLayerBuilder;
use tracing_subscriber::prelude::*;
use std::path::PathBuf;
use std::io::{self, Write};

use candle_examples::token_output_stream::TokenOutputStream;
use candle_transformers::models::mixformer::{Config, MixFormerSequentialForCausalLM as MixFormer};
use candle_transformers::models::phi::{Config as PhiConfig, Model as Phi};
use candle_transformers::models::phi3::{Config as Phi3Config, Model as Phi3};
use candle_transformers::models::quantized_mixformer::MixFormerSequentialForCausalLM as QMixFormer;

use candle::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use hf_hub::{api::sync::{Api, ApiRepo}, Repo, RepoType};
use tokenizers::Tokenizer;

const POST_PROMPT : &str = "Answer the question only. Do not discuss other irrelevant stuff.";

#[derive(Clone)]
enum Model {
    MixFormer(MixFormer),
    Phi(Phi),
    Phi3(Phi3),
    Quantized(QMixFormer),
}

struct TextGeneration {
    model: Model,
    device: Device,
    tokenizer: TokenOutputStream,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
    verbose_prompt: bool,
}

impl TextGeneration {
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: Model,
        tokenizer: Tokenizer,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        verbose_prompt: bool,
        device: &Device,
    ) -> Self {
        let logits_processor = LogitsProcessor::new(seed, temp, top_p);
        Self {
            model,
            tokenizer: TokenOutputStream::new(tokenizer),
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            verbose_prompt,
            device: device.clone(),
        }
    }

    fn run(&mut self, prompt: &str, sample_len: usize) -> Result<()> {
        use std::io::Write;
        println!("starting the inference loop");
        let tokens = self
            .tokenizer
            .tokenizer()
            .encode(prompt, true)
            .map_err(E::msg)?;
        if tokens.is_empty() {
            anyhow::bail!("Empty prompts are not supported in the phi model.")
        }
        if self.verbose_prompt {
            for (token, id) in tokens.get_tokens().iter().zip(tokens.get_ids().iter()) {
                let token = token.replace('▁', " ").replace("<0x0A>", "\n");
                println!("{id:7} -> '{token}'");
            }
        }
        let mut tokens = tokens.get_ids().to_vec();
        let mut generated_tokens = 0usize;
        let eos_token = match self.tokenizer.get_token("<|endoftext|>") {
            Some(token) => token,
            None => anyhow::bail!("cannot find the endoftext token"),
        };
        print!("{prompt}");
        std::io::stdout().flush()?;
        let start_gen = std::time::Instant::now();
        let mut pos = 0;
        for index in 0..sample_len {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = match &mut self.model {
                Model::MixFormer(m) => m.forward(&input)?,
                Model::Phi(m) => m.forward(&input)?,
                Model::Quantized(m) => m.forward(&input)?,
                Model::Phi3(m) => m.forward(&input, pos)?.i((.., 0, ..))?,
            };
            let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;
            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    &tokens[start_at..],
                )?
            };

            let next_token = self.logits_processor.sample(&logits)?;
            tokens.push(next_token);
            generated_tokens += 1;
            if next_token == eos_token {
                break;
            }
            if let Some(t) = self.tokenizer.next_token(next_token)? {
                print!("{t}");
                std::io::stdout().flush()?;
            }
            pos += context_size;
        }
        let dt = start_gen.elapsed();
        println!(
            "\n{generated_tokens} tokens generated ({:.2} token/s)",
            generated_tokens as f64 / dt.as_secs_f64(),
        );
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
enum ModelVer {
    #[value(name = "1")]
    V1,
    #[value(name = "1.5")]
    V1_5,
    #[value(name = "2")]
    V2,
    #[value(name = "3")]
    V3,
    #[value(name = "2-old")]
    V2Old,
    PuffinPhiV2,
    PhiHermes,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long, default_value_t = false)]
    cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    /// Display the token for the specified prompt.
    #[arg(long)]
    verbose_prompt: bool,

    //#[arg(long)]
    //prompt: Option<String>,

    #[arg(long)]
    mmlu_dir: Option<String>,

    /// The temperature used to generate samples.
    #[arg(long, default_value_t = 0.65)]
    temperature: f64,

    /// Nucleus sampling probability cutoff.
    #[arg(long, default_value_t = 0.7)]
    top_p: f64,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// The length of the sample to generate (in tokens).
    #[arg(long, short = 'n', default_value_t = 10000)]
    sample_len: usize,

    #[arg(long, default_value = "3")]
    model: ModelVer,

    #[arg(long)]
    revision: Option<String>,

    #[arg(long)]
    weight_file: Option<String>,

    #[arg(long)]
    tokenizer: Option<String>,

    #[arg(long, default_value_t = false)]
    quantized: bool,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,

    /// The dtype to be used for running the model, e.g. f32, bf16, or f16.
    #[arg(long)]
    dtype: Option<String>,
}
#[derive(Clone)]
struct ModelParams {
    model_ver: ModelVer,
    quantized: bool,
    revision: Option<String>,
    tokenizer: Option<String>,
    weight_file: Option<String>
}
impl ModelParams {
    fn new(model_ver: ModelVer, quantized: bool, revision: Option<String>, tokenizer: Option<String>, weight_file: Option<String>) -> Self {
        Self {
            model_ver,
            quantized,
            revision,
            tokenizer,
            weight_file
        }
    }
}

fn parse_args() -> Args {
    let args = Args::parse();
    let _guard = if args.tracing {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };
    println!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle::utils::with_avx(),
        candle::utils::with_neon(),
        candle::utils::with_simd128(),
        candle::utils::with_f16c()
    );
    println!(
        "temp: {:.2} repeat-penalty: {:.2} repeat-last-n: {}",
        args.temperature,
        args.repeat_penalty,
        args.repeat_last_n
    );

    args
}


fn initialize_api_repo(model_params: &ModelParams) -> Result<ApiRepo, ApiError> {
    println!("initializing api repo...");

    let api = Api::new()?;
    let model_id = 
        if model_params.quantized {
            "lmz/candle-quantized-phi".to_string()
        } else {
            match model_params.model_ver {
                ModelVer::V1 => "microsoft/phi-1".to_string(),
                ModelVer::V1_5 => "microsoft/phi-1_5".to_string(),
                ModelVer::V2 | ModelVer::V2Old => "microsoft/phi-2".to_string(),
                ModelVer::V3 => "microsoft/Phi-3-mini-4k-instruct".to_string(),
                ModelVer::PuffinPhiV2 | ModelVer::PhiHermes => {
                    "lmz/candle-quantized-phi".to_string()
                }
            }
        };

    let revision = match &model_params.revision {
        Some(rev) => rev.to_string(),
        None => {
            if model_params.quantized {
                "main".to_string()
            } else {
                match model_params.model_ver {
                    ModelVer::V1 => "refs/pr/8".to_string(),
                    ModelVer::V1_5 => "refs/pr/73".to_string(),
                    ModelVer::V2Old => "834565c23f9b28b96ccbeabe614dd906b6db551a".to_string(),
                    ModelVer::V2
                    | ModelVer::V3
                    | ModelVer::PuffinPhiV2
                    | ModelVer::PhiHermes => "main".to_string(),
                }
            }
        }
    };
    let repo = api.repo(Repo::with_revision(model_id, RepoType::Model, revision));

    Ok(repo)
}

struct TokenizerInfo {
    tokenizer: Tokenizer,
    filenames: Vec<PathBuf>,
}

fn initialize_tokenizer(api_repo: &ApiRepo, model_params: &ModelParams) -> Result<TokenizerInfo, E> {
    let start = std::time::Instant::now();

    let tokenizer_filename = match &model_params.tokenizer {
        Some(file) => std::path::PathBuf::from(file),
        None => match model_params.model_ver {
            ModelVer::V1
            | ModelVer::V1_5
            | ModelVer::V2
            | ModelVer::V2Old
            | ModelVer::V3 => api_repo.get("tokenizer.json")?,
            ModelVer::PuffinPhiV2 | ModelVer::PhiHermes => {
                api_repo.get("tokenizer-puffin-phi-v2.json")?
            }
        },
    };

    let filenames = match &model_params.weight_file {
        Some(weight_file) => vec![std::path::PathBuf::from(weight_file)],
        None => {
            if model_params.quantized {
                match model_params.model_ver {
                    ModelVer::V1 => vec![api_repo.get("model-v1-q4k.gguf")?],
                    ModelVer::V1_5 => vec![api_repo.get("model-q4k.gguf")?],
                    ModelVer::V2 | ModelVer::V2Old => vec![api_repo.get("model-v2-q4k.gguf")?],
                    ModelVer::PuffinPhiV2 => vec![api_repo.get("model-puffin-phi-v2-q4k.gguf")?],
                    ModelVer::PhiHermes => vec![api_repo.get("model-phi-hermes-1_3B-q4k.gguf")?],
                    ModelVer::V3 => anyhow::bail!(
                        "use the quantized or quantized-phi examples for quantized phi-v3"
                    ),
                }
            } else {
                match model_params.model_ver {
                    ModelVer::V1 | ModelVer::V1_5 => vec![api_repo.get("model.safetensors")?],
                    ModelVer::V2 | ModelVer::V2Old | ModelVer::V3 => {
                        candle_examples::hub_load_safetensors(
                            &api_repo,
                            "model.safetensors.index.json",
                        )?
                    }
                    ModelVer::PuffinPhiV2 => vec![api_repo.get("model-puffin-phi-v2.safetensors")?],
                    ModelVer::PhiHermes => vec![api_repo.get("model-phi-hermes-1_3B.safetensors")?],
                }
            }
        }
    };
    println!("retrieved the files in {:?}", start.elapsed());
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    Ok(TokenizerInfo{tokenizer, filenames})
}

struct ModelInfo {
    model: Model,
    device: Device,
}

fn load_model(
    api_repo:   &ApiRepo, 
    model_params: &ModelParams, 
    filenames:  &Vec<PathBuf>, 
    cpu:        bool, 
    dtype:      &Option<String>) 
    -> Result<ModelInfo, E> {
    let start = std::time::Instant::now();
    let config = || match model_params.model_ver {
        ModelVer::V1 => Config::v1(),
        ModelVer::V1_5 => Config::v1_5(),
        ModelVer::V2 | ModelVer::V2Old => Config::v2(),
        ModelVer::PuffinPhiV2 => Config::puffin_phi_v2(),
        ModelVer::PhiHermes => Config::phi_hermes_1_3b(),
        ModelVer::V3 => {
            panic!("use the quantized or quantized-phi examples for quantized phi-v3")
        }
    };
    let device = candle_examples::device(cpu)?;
    let model = if model_params.quantized {
        let config = config();
        let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(
            &filenames[0],
            &device,
        )?;
        let model = match model_params.model_ver {
            ModelVer::V2 | ModelVer::V2Old => QMixFormer::new_v2(&config, vb)?,
            _ => QMixFormer::new(&config, vb)?,
        };
        Model::Quantized(model)
    } else {
        let dtype = match &dtype {
            Some(dtype) => std::str::FromStr::from_str(&dtype)?,
            None => {
                if model_params.model_ver == ModelVer::V3 && device.is_cuda() {
                    DType::BF16
                } else {
                    DType::F32
                }
            }
        };
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };
        match model_params.model_ver {
            ModelVer::V1 | ModelVer::V1_5 | ModelVer::V2 => {
                let config_filename = api_repo.get("config.json")?;
                let config = std::fs::read_to_string(config_filename)?;
                let config: PhiConfig = serde_json::from_str(&config)?;
                let phi = Phi::new(&config, vb)?;
                Model::Phi(phi)
            }
            ModelVer::V3 => {
                let config_filename = api_repo.get("config.json")?;
                let config = std::fs::read_to_string(config_filename)?;
                let config: Phi3Config = serde_json::from_str(&config)?;
                let phi3 = Phi3::new(&config, vb)?;
                Model::Phi3(phi3)
            }
            ModelVer::V2Old => {
                let config = config();
                Model::MixFormer(MixFormer::new_v2(&config, vb)?)
            }
            ModelVer::PhiHermes | ModelVer::PuffinPhiV2 => {
                let config = config();
                Model::MixFormer(MixFormer::new(&config, vb)?)
            }
        }
    };
    println!("loaded the model in {:?}", start.elapsed());

    Ok(ModelInfo{model, device})
}

struct PromptParams {
    sample_len: usize,
    seed: u64,
    temperature: f64,
    top_p: f64,
    repeat_penalty: f32,
    repeat_last_n: usize,
    verbose_prompt: bool,
}

impl PromptParams {
    fn new(
        sample_len: usize,
        seed: u64,
        temperature: f64,
        top_p: f64,
        repeat_penalty: f32,
        repeat_last_n: usize,
        verbose_prompt: bool
        ) -> Self {
        Self {
            sample_len: sample_len,
            seed: seed,
            temperature: temperature,
            top_p: top_p,
            repeat_penalty: repeat_penalty,
            repeat_last_n: repeat_last_n,
            verbose_prompt: verbose_prompt,
        }
    }
}

fn submit_prompt(
    prompt:         &str, 
    prompt_params:  &PromptParams,
    model_info:     &ModelInfo,
    tokenizer:      &Tokenizer)
    -> Result<()> {
    let mut pipeline = TextGeneration::new(
        model_info.model.clone(),
        tokenizer.clone(),
        prompt_params.seed,
        Some(prompt_params.temperature),
        Some(prompt_params.top_p),
        prompt_params.repeat_penalty,
        prompt_params.repeat_last_n,
        prompt_params.verbose_prompt,
        &model_info.device,
    );

    pipeline.run(prompt, prompt_params.sample_len)?;

    Ok(())
}


fn main() -> Result<()> {
    let args = parse_args();

    let model_params = ModelParams::new(args.model, args.quantized, args.revision.clone(), args.tokenizer.clone(), args.weight_file.clone());
    let api_repo = initialize_api_repo(&model_params)?;

    let tokenizer_info = initialize_tokenizer(&api_repo, &model_params)?;

    let model_info = load_model(
        &api_repo,
        &model_params,
        &tokenizer_info.filenames,
        args.cpu,
        &args.dtype)?;

    //
    // Prompt the model
    //
    let prompt_params = PromptParams::new(
        args.sample_len,
        args.seed,
        args.temperature,
        args.top_p,
        args.repeat_penalty,
        args.repeat_last_n,
        args.verbose_prompt
    );
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

                submit_prompt(&prompt, &prompt_params, &model_info, &tokenizer_info.tokenizer)?;
            }
            Err(error) => {
                // Handle any errors that occur during reading
                eprintln!("Error reading input: {}", error);
                break;
            }
        }
    }

    Ok(())
}

#[allow(dead_code)]
fn mmlu<P: AsRef<std::path::Path>>(
    mut model: Model,
    tokenizer: Tokenizer,
    device: &Device,
    mmlu_dir: P,
) -> anyhow::Result<()> {
    for dir_entry in mmlu_dir.as_ref().read_dir()?.flatten() {
        let dir_entry = dir_entry.path();
        let theme = match dir_entry.file_stem().and_then(|v| v.to_str()) {
            None => "".to_string(),
            Some(v) => match v.strip_suffix("_test") {
                None => v.replace('_', " "),
                Some(v) => v.replace('_', " "),
            },
        };
        if dir_entry.extension().as_ref().and_then(|v| v.to_str()) != Some("csv") {
            continue;
        }
        println!("reading {dir_entry:?}");
        let dir_entry = std::fs::File::open(dir_entry)?;
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(false)
            .from_reader(dir_entry);
        let token_a = tokenizer.token_to_id("A").unwrap();
        let token_b = tokenizer.token_to_id("B").unwrap();
        let token_c = tokenizer.token_to_id("C").unwrap();
        let token_d = tokenizer.token_to_id("D").unwrap();
        for row in reader.records() {
            let row = match row {
                Err(_) => continue,
                Ok(row) => row,
            };
            if row.len() < 5 {
                continue;
            }
            let question = row.get(0).unwrap();
            let answer_a = row.get(1).unwrap();
            let answer_b = row.get(2).unwrap();
            let answer_c = row.get(3).unwrap();
            let answer_d = row.get(4).unwrap();
            let answer = row.get(5).unwrap();
            let prompt = format!(
                    "{} {theme}.\n{question}\nA. {answer_a}\nB. {answer_b}\nC. {answer_c}\nD. {answer_d}\nAnswer:\n",
                    "The following are multiple choice questions (with answers) about"
                );
            let tokens = tokenizer.encode(prompt.as_str(), true).map_err(E::msg)?;
            let tokens = tokens.get_ids().to_vec();
            let input = Tensor::new(tokens, device)?.unsqueeze(0)?;
            let logits = match &mut model {
                Model::MixFormer(m) => {
                    m.clear_kv_cache();
                    m.forward(&input)?
                }
                Model::Phi(m) => {
                    m.clear_kv_cache();
                    m.forward(&input)?
                }
                Model::Phi3(m) => {
                    m.clear_kv_cache();
                    m.forward(&input, 0)?
                }
                Model::Quantized(m) => {
                    m.clear_kv_cache();
                    m.forward(&input)?
                }
            };
            let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;
            let logits_v: Vec<f32> = logits.to_vec1()?;
            let pr_a = logits_v[token_a as usize];
            let pr_b = logits_v[token_b as usize];
            let pr_c = logits_v[token_c as usize];
            let pr_d = logits_v[token_d as usize];
            let model_answer = if pr_a > pr_b && pr_a > pr_c && pr_a > pr_d {
                "A"
            } else if pr_b > pr_c && pr_b > pr_d {
                "B"
            } else if pr_c > pr_d {
                "C"
            } else {
                "D"
            };

            println!("{prompt}\n -> {model_answer} vs {answer}");
        }
    }
    Ok(())
}
