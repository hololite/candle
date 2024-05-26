cargo run --example quantized-phi --features cuda -- --prompt "which inference model are you based on? answer the que
stion only, don't discuss other stuff"

cargo run --example quantized --features cuda -- --which llama3-8b --prompt interactive
cargo run --example quantized --features cuda -- --which phi3      --prompt interactive

cargo run --example phi --features cuda
cargo run --example llama --features cuda 
