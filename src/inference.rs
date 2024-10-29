use directories::ProjectDirs;
use mistralrs::DefaultSchedulerMethod;
use mistralrs::Device;
use mistralrs::DeviceMapMetadata;
use mistralrs::GGUFLoaderBuilder;
use mistralrs::GGUFSpecificConfig;
use mistralrs::MistralRs;
use mistralrs::MistralRsBuilder;
use mistralrs::ModelDType;
use mistralrs::SchedulerConfig;
use mistralrs::TokenSource;
use std::fs::create_dir_all;
use std::fs::File;
use std::io;
use std::io::Write;
use std::sync::Arc;
use tokio::runtime::Runtime;

const MODEL: &str = "Humanish-LLama3-8B-Instruct-Q4_K_M.gguf";
pub const SYSTEM_PROMPT: &str = r#"
You are *The Quirky Scientist*—a warm, highly knowledgeable AI with a playful edge and a passion for science. You're a bit nerdy, love quirky facts, and explain complex ideas with fun analogies. Your tone is friendly, upbeat, and slightly whimsical, making even challenging topics feel accessible. Answer should be short, with a minimal number of words and witty."#;
async fn download_model() -> anyhow::Result<()> {
    println!("Downloading Model");
    let dir = ProjectDirs::from("com", "termi-talk", "rhea").unwrap();
    create_dir_all(dir.data_dir())?;
    let path = dir.data_dir().join(MODEL);
    let mut response = reqwest::get("https://huggingface.co/bartowski/Humanish-LLama3-8B-Instruct-GGUF/resolve/main/Humanish-LLama3-8B-Instruct-Q4_K_M.gguf?download=true").await?;

    if response.status().is_success() {
        let total_size = response.content_length().unwrap_or(0);
        let mut file = File::create(path)?;
        let mut downloaded: u64 = 0;

        println!("Downloading {} bytes...", total_size);
        while let Some(chunk) = response.chunk().await? {
            file.write_all(&chunk)?;
            downloaded += chunk.len() as u64;

            if total_size > 0 {
                let percentage = (downloaded as f64 / total_size as f64) * 100.0;
                print!(
                    "\rDownloaded: {} of {} bytes ({:.2}%)",
                    downloaded, total_size, percentage
                );
            } else {
                print!("\rDownloaded: {} bytes", downloaded);
            }
            io::stdout().flush()?;
        }
        println!("File downloaded successfully to {:?}", dir.data_dir());
    } else {
        eprintln!("Failed to download file: {}", response.status());
    }
    Ok(())
}

pub fn load_model() -> anyhow::Result<Arc<MistralRs>> {
    let dir = ProjectDirs::from("com", "termi-talk", "rhea").unwrap();
    let path = dir.data_dir().join(MODEL);
    if !path.exists() {
        let rt = Runtime::new().unwrap();
        rt.block_on(async {
            let _ = download_model().await;
        });
    }

    let loader = GGUFLoaderBuilder::new(
        None,
        None,
        dir.data_dir().to_str().unwrap().to_string(),
        vec![MODEL.to_string()],
        GGUFSpecificConfig {
            prompt_batchsize: None,
            topology: None,
        },
    )
    .build();

    let pipeline = loader.load_model_from_hf(
        None,
        TokenSource::CacheToken,
        &ModelDType::Auto,
        &Device::cuda_if_available(0).unwrap(),
        false,
        DeviceMapMetadata::dummy(),
        None,
        None,
    )?;

    Ok(MistralRsBuilder::new(
        pipeline,
        SchedulerConfig::DefaultScheduler {
            method: DefaultSchedulerMethod::Fixed(5.try_into().unwrap()),
        },
    )
    .build())
}

pub fn wrap_text(text: String, max_width: usize) -> Vec<String> {
    let mut lines = Vec::new();
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut current_line = String::new();

    for word in words {
        if current_line.len() + word.len() + 12 > max_width {
            lines.push(current_line.trim().to_string());
            current_line = String::new();
        }
        current_line.push_str(word);
        current_line.push(' ');
    }

    if !current_line.is_empty() {
        lines.push(current_line.trim().to_string());
    }

    lines
}
