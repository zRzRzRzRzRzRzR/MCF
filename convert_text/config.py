import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GLM_API_KEY = os.getenv('GLM_API_KEY')
    if not GLM_API_KEY:
        raise ValueError("Please set the GLM_API_KEY environment variable in the .env file.")
    
    GLM_BASE_URL = 'https://open.bigmodel.cn/api/paas/v4/'
    GLM_MODEL = 'glm-4.5'

    CONFIDENCE_THRESHOLD = 0.7
    MAX_TOKENS = 2000
    TEMPERATURE = 0.1

    INPUT_DIR = './input'
    OUTPUT_DIR = './output'
    LOG_DIR = './logs'

    MAX_RETRIES = 3
    REQUEST_TIMEOUT = 30
    RATE_LIMIT_DELAY = 0.1