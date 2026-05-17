from pydantic_settings import basesettings


class settings(basesettings):
    # api configuration parameters
    app_name: str = "stocky tft engine api"
    version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = 8000

    # model paths
    model_checkpoint: str = "checkpoints/best_tft_model.pt"

    # logging
    log_level: str = "info"

    class config:
        env_file = ".env"


app_settings = settings()