import os
import json
import hashlib
import datetime
import argparse
from pathlib import Path
from tqdm import tqdm
import whisper
import ffmpeg

class AudioTranscriber:
    def __init__(self, audio_dir, transcript_dir, language=None):
        self.audio_dir = Path(audio_dir)
        self.transcript_dir = Path(transcript_dir)
        self.processed_files_json = self.transcript_dir / "processed_files.json"
        self.language = language  # Language code: 'en', 'zh', 'id', or None for auto-detection
        
        print(f"Audio directory: {self.audio_dir}")
        print(f"Transcript directory: {self.transcript_dir}")
        print(f"Language: {self.language if self.language else 'Auto-detect'}")
        print("Loading Whisper model...")
        self.model = whisper.load_model("large")
        print("Large Whisper model loaded!")
        self.processed_files = self._load_processed_files()

        # Define file extensions
        self.audio_extensions = {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a'}
        self.video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv'}

    def _load_processed_files(self):
        """Load the processed files record or create if doesn't exist."""
        if self.processed_files_json.exists():
            with open(self.processed_files_json, 'r') as f:
                return json.load(f)
        return {}

    def _save_processed_files(self):
        """Save the processed files record."""
        with open(self.processed_files_json, 'w') as f:
            json.dump(self.processed_files, f, indent=4)

    def _calculate_file_hash(self, file_path):
        """Calculate MD5 hash of a file."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _needs_processing(self, file_path, file_hash):
        """Check if file needs to be processed based on hash."""
        if str(file_path) not in self.processed_files:
            return True
        return self.processed_files[str(file_path)]["hash"] != file_hash

    def _extract_audio_from_video(self, video_path):
        """Extract audio from video file using ffmpeg."""
        # Create a temporary audio file in the same directory
        audio_path = video_path.with_suffix('.mp3')
        
        try:
            # Use ffmpeg-python to extract audio
            print(f"Attempting to extract audio from: {video_path}")
            (
                ffmpeg
                .input(str(video_path))
                .output(str(audio_path), acodec='libmp3lame', q=4)
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            print(f"Audio extracted successfully to: {audio_path}")
            return audio_path
        except ffmpeg.Error as e:
            print(f"Error extracting audio from {video_path}: {e.stderr.decode()}")
            raise

    def _find_media_files(self):
        """Find all audio and video files in the directory."""
        # List all files in the audio directory
        print("Listing all files in the audio directory:")
        audio_files = []
        video_files = []

        for file in self.audio_dir.iterdir():
            print(f"Found file: {file}")
            # Skip .gitkeep and hidden files
            if file.name.startswith('.') or file.name == '.gitkeep':
                continue

            # Check file extension
            if file.suffix.lower() in self.audio_extensions:
                audio_files.append(file)
                print(f"Found audio file: {file}")
            elif file.suffix.lower() in self.video_extensions:
                video_files.append(file)
                print(f"Found video file: {file}")

        return audio_files, video_files

    def transcribe_audio(self):
        """Transcribe all audio and video files in the directory that need processing."""
        # Find audio and video files
        audio_files, video_files = self._find_media_files()
        print(f"Found {len(audio_files)} audio files")
        print(f"Found {len(video_files)} video files")

        # Extract audio from video files
        for video_file in video_files:
            try:
                audio_file = self._extract_audio_from_video(video_file)
                audio_files.append(audio_file)
            except Exception as e:
                print(f"Error processing {video_file}: {str(e)}")

        # Transcribe files
        for audio_file in tqdm(audio_files, desc="Processing audio files"):
            # Make paths relative
            rel_audio_file = audio_file.relative_to(Path(__file__).parent.parent)
            
            file_hash = self._calculate_file_hash(audio_file)
            
            if not self._needs_processing(rel_audio_file, file_hash):
                print(f"Skipping {audio_file.name} - already processed")
                continue

            print(f"Transcribing {audio_file.name}...")
            try:
                # Show spinner during transcription
                with tqdm(total=0, desc="Transcribing", bar_format='{desc}: {elapsed}') as pbar:
                    # Add language parameter if specified
                    transcribe_options = {}
                    if self.language:
                        transcribe_options["language"] = self.language
                    
                    result = self.model.transcribe(str(audio_file), **transcribe_options)
                    transcript = result["text"].strip()
                
                # Save transcription with relative paths
                transcript_file = self.transcript_dir / f"{audio_file.stem}.txt"
                rel_transcript_file = transcript_file.relative_to(Path(__file__).parent.parent)
                
                with open(transcript_file, "w", encoding="utf-8") as f:
                    f.write(transcript)

                # Update processed files record with relative paths
                self.processed_files[str(rel_audio_file)] = {
                    "hash": file_hash,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "transcript_file": str(rel_transcript_file),
                    "language": self.language if self.language else "auto"
                }
                
                # Save after each successful transcription
                self._save_processed_files()
                
            except Exception as e:
                print(f"Error processing {audio_file.name}: {str(e)}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Transcribe audio files with Whisper')
    parser.add_argument('--language', '-l', choices=['en', 'zh', 'id'], 
                        help='Language of audio (en=English, zh=Chinese, id=Bahasa Indonesia)')
    args = parser.parse_args()

    # Define paths
    base_dir = Path(__file__).parent.parent
    audio_dir = base_dir / "data" / "raw" / "audio"
    transcript_dir = base_dir / "data" / "processed" / "transcript"

    # Create directories if they don't exist
    audio_dir.mkdir(parents=True, exist_ok=True)
    transcript_dir.mkdir(parents=True, exist_ok=True)

    # Initialize and run transcriber
    transcriber = AudioTranscriber(audio_dir, transcript_dir, args.language)
    transcriber.transcribe_audio()

if __name__ == "__main__":
    main()
