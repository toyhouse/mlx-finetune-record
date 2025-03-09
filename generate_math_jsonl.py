import os
from typing import List
import json
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_log, after_log
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from jsonschema import validate, ValidationError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('math_qa_generation.log')
    ]
)

def setup_gemini():
    """Setup Gemini model"""
    load_dotenv()
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-pro-exp-02-05",
        temperature=0.7,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    return llm

def create_prompt_template() -> ChatPromptTemplate:
    """Create the prompt template for generating JSONL content"""
    template = """You are an AI assistant that helps convert math teaching transcripts into JSONL format.
    Given the following transcript of a math teaching video:
    1. Correct any typos and unclear explanations in Indonesian
    2. Use proper Indonesian mathematical terms
    3. Make sure the explanation is clear and step-by-step
    4. Keep the mathematical method (Gasing) intact
    5. Format numbers clearly (avoid using hyphens like 10-2, instead use "10 ditambah 2")
    
    Transcript:
    {transcript}
    
    First, correct the transcript to proper Indonesian, then generate the content in this exact format:
    {{"text": "You are a math teacher using the Gasing method\\n\\nHuman: [generated question in Indonesian]\\n\\nAssistant: [corrected explanation with proper Indonesian]"}}
    
    The explanation should:
    - Use proper Indonesian spelling (e.g., "delapan" not "dilapan")
    - Have clear step-by-step instructions
    - Use proper mathematical terms
    - Maintain a clear flow of explanation
    - End with a clear conclusion
    """
    
    return ChatPromptTemplate.from_template(template)

class InvalidJSONLFormatError(Exception):
    """Raised when the generated JSONL is not properly formatted"""
    pass

def validate_jsonl_format(content: str, transcript_path: str) -> bool:
    """Validate if the content follows proper JSONL format with required fields"""
    # Define the expected schema for each JSON line
    schema = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                # More flexible pattern that allows any whitespace between sections
                "pattern": ".*You are a math teacher using the Gasing method.*Human:.*Assistant:.*"
            }
        },
        "required": ["text"]
    }
    
    try:
        # Parse the content as JSON
        json_obj = json.loads(content)
        # Validate against schema
        validate(instance=json_obj, schema=schema)
        
        # Additional format checks
        text_content = json_obj.get('text', '')
        if not text_content:
            logging.error(f"Empty text field in {transcript_path}")
            return False
            
        # Check for proper section markers
        if 'Human:' not in text_content or 'Assistant:' not in text_content:
            logging.error(f"Missing Human/Assistant markers in {transcript_path}")
            return False
            
        # Check for minimum content length in each section
        human_part = text_content.split('Human:')[1].split('Assistant:')[0]
        assistant_part = text_content.split('Assistant:')[1]
        
        if len(human_part.strip()) < 10:
            logging.error(f"Human question too short in {transcript_path}")
            return False
            
        if len(assistant_part.strip()) < 50:
            logging.error(f"Assistant answer too short in {transcript_path}")
            return False
        
        logging.info(f"Successfully validated content for {transcript_path}")
        return True
        
    except json.JSONDecodeError as e:
        logging.error(f"JSON parsing error in {transcript_path}: {str(e)}")
        return False
    except ValidationError as e:
        logging.error(f"Schema validation error in {transcript_path}: {str(e)}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error validating {transcript_path}: {str(e)}")
        return False

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=10, max=120),  # Increased wait times
    retry=retry_if_exception_type((Exception, InvalidJSONLFormatError)),
    before=before_log(logging.getLogger(), logging.INFO),
    after=after_log(logging.getLogger(), logging.INFO)
)
def process_transcript(transcript_path: str, llm, prompt_template: ChatPromptTemplate) -> str:
    """Process a single transcript with retry mechanism and format validation"""
    with open(transcript_path, 'r', encoding='utf-8') as f:
        transcript = f.read().strip()
    
    try:
        chain = prompt_template | llm
        result = chain.invoke({"transcript": transcript})
        content = result.content.replace('```jsonl', '').replace('```', '').replace('json', '').strip()
        
        # Validate JSONL format with detailed checks
        if not validate_jsonl_format(content, transcript_path):
            error_msg = f"Generated content for {transcript_path} is not in valid JSONL format"
            logging.error(error_msg)
            raise InvalidJSONLFormatError(error_msg)
        
        return content
    except Exception as e:
        print(f"Attempt failed for {transcript_path}: {str(e)}")
        raise

def process_all_transcripts(transcript_dir: str, output_file: str):
    """Process all transcript files and generate JSONL"""
    llm = setup_gemini()
    prompt_template = create_prompt_template()
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for filename in os.listdir(transcript_dir):
            if filename.endswith('.txt'):
                transcript_path = os.path.join(transcript_dir, filename)
                try:
                    print(f"Processing: {filename}")
                    jsonl_content = process_transcript(transcript_path, llm, prompt_template)
                    # Ensure the content ends with exactly one newline
                    jsonl_content = jsonl_content.strip() + '\n'
                    out_f.write(jsonl_content)
                    print(f"Successfully processed: {filename}")
                    # Increased delay to avoid rate limits
                    time.sleep(15)  # 15-second delay between files
                except Exception as e:
                    logging.error(f"Failed to process {filename} after all retries: {str(e)}")
                    # Write error to a separate error log file
                    with open('failed_transcripts.log', 'a') as error_f:
                        error_f.write(f"{filename}: {str(e)}\n")
                    continue

def main():
    # Get script directory for relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Navigate from script location to transcript directory using relative path
    transcript_dir = os.path.normpath(os.path.join(script_dir, "../../../Docs/to-do-plan/data/processed/transcript"))
    # Output relative to script location
    output_file = os.path.join(script_dir, "math_qa.jsonl")
    
    process_all_transcripts(transcript_dir, output_file)
    print(f"JSONL file generated at: {output_file}")

if __name__ == "__main__":
    main()