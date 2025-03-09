import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import os

class KidMathEvaluator:
    def __init__(self, model_name, use_ollama=True):
        """Initialize the evaluator with a model."""
        self.model_name = model_name
        self.use_ollama = use_ollama
        
        # Categories of problems to test
        self.categories = [
            "addition", "subtraction", "multiplication", 
            "division", "fractions", "word_problems"
        ]
        
        # Grade levels to test (1-5)
        self.grade_levels = [1, 2, 3, 4, 5]
        
        # Educational quality metrics
        self.quality_metrics = [
            "explanation_clarity", 
            "step_by_step_guidance",
            "child_friendly_language",
            "visual_representation",
            "positive_reinforcement"
        ]
        
        print(f"Initialized evaluator for Ollama model: {model_name}")
    
    def load_test_data(self, data_path):
        """Load grade-appropriate test problems."""
        with open(data_path, 'r') as f:
            self.test_data = json.load(f)
        
        # Validate data structure
        required_fields = ["problem", "answer", "grade", "category"]
        for item in self.test_data:
            if not all(field in item for field in required_fields):
                raise ValueError(f"Test data missing required fields: {required_fields}")
        
        print(f"Loaded {len(self.test_data)} test problems across {len(self.grade_levels)} grade levels")
    
    def generate_response(self, prompt, max_new_tokens=512):
        """Generate a response from the model using Ollama API."""
        try:
            # Use the completion endpoint instead of generate for non-streaming response
            response = requests.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": self.model_name,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": max_new_tokens
                    }
                },
                timeout=60
            )
            response.raise_for_status()
            
            # Extract the response content
            result = response.json()
            if "message" in result and "content" in result["message"]:
                return result["message"]["content"].strip()
            else:
                # Fallback to old API format if needed
                return result.get("response", "").strip()
                
        except Exception as e:
            print(f"Error generating response from Ollama: {e}")
            # Return a placeholder instead of the error to avoid breaking the evaluation
            return "Error generating response"
    
    def evaluate_correctness(self):
        """Evaluate whether the model produces correct answers."""
        results = []
        
        for item in tqdm(self.test_data, desc="Evaluating correctness"):
            # Format a kid-friendly prompt
            prompt = f"""
I'm learning math and need help with this problem:

{item['problem']}

Can you help me solve this step by step?
"""
            response = self.generate_response(prompt)
            
            # Simple accuracy check - does the response contain the correct answer?
            correct = str(item['answer']) in response
            
            results.append({
                "model": self.model_name,  # Add the model name to each result
                "problem": item['problem'],
                "expected": item['answer'],
                "response": response,
                "correct": correct,
                "grade": item['grade'],
                "category": item['category']
            })
        
        self.correctness_results = pd.DataFrame(results)
        return self.correctness_results
    
    def evaluate_educational_quality(self, sample_size=50):
        """Evaluate educational quality metrics using a rubric."""
        # Sample problems across grades and categories
        sampled_data = self._sample_diverse_problems(sample_size)
        
        results = []
        rubric = self._get_educational_quality_rubric()
        
        for item in tqdm(sampled_data, desc="Evaluating educational quality"):
            prompt = f"""
I'm in grade {item['grade']} and learning math. 
Can you help me understand and solve this problem?

{item['problem']}
"""
            response = self.generate_response(prompt)
            
            # Manual scoring would happen here in a real implementation
            # For automation, we could use a second LLM to evaluate the first
            quality_scores = self._score_educational_quality(response, rubric)
            
            results.append({
                "problem": item['problem'],
                "grade": item['grade'],
                "category": item['category'],
                "response": response,
                **quality_scores
            })
        
        self.quality_results = pd.DataFrame(results)
        return self.quality_results
    
    def _sample_diverse_problems(self, sample_size):
        """Sample problems ensuring diversity across grades and categories."""
        sampled = []
        
        # Stratified sampling by grade and category
        for grade in self.grade_levels:
            grade_problems = [p for p in self.test_data if p['grade'] == grade]
            
            for category in self.categories:
                category_problems = [p for p in grade_problems if p['category'] == category]
                
                # Take up to n/30 problems from each grade/category combination
                # (5 grades × 6 categories = 30 combinations)
                n_to_sample = max(1, sample_size // 30)
                if category_problems:
                    sampled.extend(np.random.choice(
                        category_problems, 
                        size=min(n_to_sample, len(category_problems)),
                        replace=False
                    ).tolist())
        
        return sampled
    
    def _get_educational_quality_rubric(self):
        """Define a rubric for scoring educational quality."""
        return {
            "explanation_clarity": "Does the model explain concepts clearly at the child's level?",
            "step_by_step_guidance": "Does the model break down the problem into manageable steps?",
            "child_friendly_language": "Does the model use age-appropriate language and examples?",
            "visual_representation": "Does the model use textual visualization techniques?",
            "positive_reinforcement": "Does the model provide encouragement and positive feedback?"
        }
    
    def _score_educational_quality(self, response, rubric):
        """
        Score a response on educational quality metrics.
        
        In a real implementation, this would be done by:
        1. Human evaluators using the rubric
        2. A second LLM evaluating the first LLM's response
        3. Automated heuristics
        
        For this example, we'll use simple heuristics.
        """
        scores = {}
        
        # Simple heuristic scoring (in practice, use more sophisticated methods)
        scores["explanation_clarity"] = self._score_clarity(response)
        scores["step_by_step_guidance"] = self._score_steps(response)
        scores["child_friendly_language"] = self._score_child_friendly(response)
        scores["visual_representation"] = self._score_visual(response)
        scores["positive_reinforcement"] = self._score_positivity(response)
        
        return scores
    
    def _score_clarity(self, text):
        """Score explanation clarity based on simple heuristics."""
        # In practice, this would be much more sophisticated
        clarity_markers = [
            "means", "is like", "think of it as", "imagine", 
            "for example", "in other words"
        ]
        return min(5, sum(marker in text.lower() for marker in clarity_markers))
    
    def _score_steps(self, text):
        """Score step-by-step guidance."""
        # Count numbered steps, "first", "then", "next", etc.
        step_markers = ["step", "first", "second", "third", "next", "then", "finally"]
        step_count = sum(marker in text.lower() for marker in step_markers)
        return min(5, step_count)
    
    def _score_child_friendly(self, text):
        """Score child-friendly language use."""
        friendly_markers = ["let's", "together", "fun", "cool", "awesome", "great job"]
        return min(5, sum(marker in text.lower() for marker in friendly_markers))
    
    def _score_visual(self, text):
        """Score visual representation (ASCII art, diagrams, etc.)."""
        visual_markers = ["|", "+---+", "[]", "()", "-->", "*", "#"]
        return min(5, sum(marker in text for marker in visual_markers))
    
    def _score_positivity(self, text):
        """Score positive reinforcement."""
        positive_markers = ["good job", "well done", "great", "excellent", "you're doing great"]
        return min(5, sum(marker in text.lower() for marker in positive_markers))
    
    def visualize_results(self):
        """Create visualizations of the evaluation results."""
        if not hasattr(self, 'correctness_results'):
            raise ValueError("Run evaluate_correctness() first")
        
        # Create a results directory
        import os
        os.makedirs("results", exist_ok=True)
        
        # Accuracy by grade and category
        plt.figure(figsize=(12, 6))
        accuracy_by_grade = self.correctness_results.groupby('grade')['correct'].mean()
        sns.barplot(x=accuracy_by_grade.index, y=accuracy_by_grade.values)
        plt.title("Accuracy by Grade Level")
        plt.xlabel("Grade")
        plt.ylabel("Accuracy")
        plt.savefig("results/accuracy_by_grade.png")
        
        # Accuracy by category
        plt.figure(figsize=(12, 6))
        accuracy_by_category = self.correctness_results.groupby('category')['correct'].mean()
        sns.barplot(x=accuracy_by_category.index, y=accuracy_by_category.values)
        plt.title("Accuracy by Problem Category")
        plt.xlabel("Category")
        plt.ylabel("Accuracy")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("results/accuracy_by_category.png")
        
        # If quality results are available
        if hasattr(self, 'quality_results'):
            # Educational quality radar chart
            plt.figure(figsize=(10, 10))
            quality_means = self.quality_results[self.quality_metrics].mean()
            
            # Create the radar chart
            angles = np.linspace(0, 2*np.pi, len(self.quality_metrics), endpoint=False)
            angles = np.concatenate((angles, [angles[0]]))  # Close the loop
            
            values = quality_means.values.tolist()
            values = values + [values[0]]  # Close the loop
            
            ax = plt.subplot(111, polar=True)
            ax.plot(angles, values, 'o-', linewidth=2)
            ax.fill(angles, values, alpha=0.25)
            ax.set_thetagrids(angles[:-1] * 180/np.pi, self.quality_metrics)
            ax.set_ylim(0, 5)
            plt.title("Educational Quality Assessment")
            plt.savefig("results/educational_quality.png")
            
        return "Visualizations saved to results directory"
    
    def generate_report(self):
        """Generate a comprehensive evaluation report."""
        if not hasattr(self, 'correctness_results'):
            raise ValueError("Run evaluate_correctness() first")
        
        report = {
            "overall_accuracy": self.correctness_results['correct'].mean(),
            "accuracy_by_grade": self.correctness_results.groupby('grade')['correct'].mean().to_dict(),
            "accuracy_by_category": self.correctness_results.groupby('category')['correct'].mean().to_dict(),
            "sample_failures": self.correctness_results[~self.correctness_results['correct']].head(5)[['problem', 'expected', 'response']].to_dict('records')
        }
        
        if hasattr(self, 'quality_results'):
            report["educational_quality"] = {
                metric: self.quality_results[metric].mean()
                for metric in self.quality_metrics
            }
        
        with open("results/evaluation_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        return report

def get_ollama_models():
    """Get list of available models from Ollama and their sizes."""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            # Sort models by size (smallest first)
            models.sort(key=lambda x: x.get("size", float("inf")))
            return models
        else:
            print(f"Error fetching models: {response.status_code}")
            return []
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        return []

def get_models_from_config(config_path="config.txt"):
    """Load list of models to test from a config file."""
    try:
        with open(config_path, 'r') as f:
            models = [line.strip() for line in f.readlines() if line.strip() and not line.startswith('#')]
        return models
    except FileNotFoundError:
        # Create default config file if it doesn't exist
        default_models = ["llama3", "qwq", "phi4", "MathTutor", "s1k_init"]
        os.makedirs(os.path.dirname(config_path) if os.path.dirname(config_path) else ".", exist_ok=True)
        with open(config_path, 'w') as f:
            f.write("# List of models to test (one per line)\n")
            f.write("# Lines starting with # are ignored\n\n")
            f.write("\n".join(default_models))
        return default_models

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate kid-friendly math LLM using Ollama")
    parser.add_argument("--model", type=str, help="Ollama model name (single model)")
    parser.add_argument("--test_data", type=str, default="/Users/bkoo/Documents/Development/AIProjects/mlx-finetune-record/benchmarks/kfa_benchmark/data.json", 
                        help="Path to test data JSON")
    parser.add_argument("--smallest", action="store_true", help="Use the three smallest models")
    parser.add_argument("--all", action="store_true", help="Test all available models")
    parser.add_argument("--config", type=str, default="config.txt", help="Path to config file with models list")
    
    args = parser.parse_args()
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Check if Ollama is running
    try:
        requests.get("http://localhost:11434/api/tags", timeout=5)
    except requests.exceptions.ConnectionError:
        print("Ollama is not running. Please start Ollama first.")
        print("You can start it with: ollama serve")
        exit(1)
        
    # Get models to test
    models_to_test = []
    if args.all:
        # Get all available models
        all_models = get_ollama_models()
        if not all_models:
            print("No models found in Ollama. Please pull some models first.")
            print("Example: ollama pull tinyllama")
            exit(1)
        
        models_to_test = [model["name"] for model in all_models]
        print(f"Testing all {len(models_to_test)} available models: {', '.join(models_to_test)}")
    elif args.smallest:
        # Get the three smallest models
        all_models = get_ollama_models()
        if not all_models:
            print("No models found in Ollama. Please pull some models first.")
            print("Example: ollama pull tinyllama")
            exit(1)
        
        models_to_test = [model["name"] for model in all_models[:3]]
        print(f"Testing the three smallest models: {', '.join(models_to_test)}")
    elif args.model:
        models_to_test = [args.model]
    else:
        # Default: load from config file
        models_to_test = get_models_from_config(args.config)
        print(f"Testing models from config: {', '.join(models_to_test)}")
    
    # Run evaluation for each model
    all_results = []
    
    for model_name in models_to_test:
        print(f"\n{'='*50}")
        print(f"Evaluating model: {model_name}")
        print(f"{'='*50}")
        
        evaluator = KidMathEvaluator(model_name)
        evaluator.load_test_data(args.test_data)
        
        print("Evaluating correctness...")
        correctness_results = evaluator.evaluate_correctness()
        all_results.append(correctness_results)
        
        print(f"Overall accuracy: {correctness_results['correct'].mean():.2f}")
        
        print("\nEvaluating educational quality...")
        quality_results = evaluator.evaluate_educational_quality(sample_size=10)
        
        print("\nGenerating visualizations...")
        evaluator.visualize_results()
        
        print("\nGenerating report...")
        report = evaluator.generate_report()
        
    # Combine all results for comparison
    if len(all_results) > 1:
        combined_results = pd.concat(all_results)
        
        # Create comparison visualizations
        plt.figure(figsize=(12, 6))
        comparison = combined_results.groupby('model')['correct'].mean()
        sns.barplot(x=comparison.index, y=comparison.values)
        plt.title("Model Accuracy Comparison")
        plt.xlabel("Model")
        plt.ylabel("Accuracy")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("results/model_comparison.png")
        
        # Save combined results
        combined_results.to_csv("results/combined_results.csv", index=False)
    
    print("Evaluation complete! Results saved to the 'results' directory.")