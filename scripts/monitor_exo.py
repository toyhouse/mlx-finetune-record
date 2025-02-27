"""
Script to monitor EXO performance during parallel processing.
"""
import argparse
import subprocess
import time
import os
import json
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description="Monitor EXO performance during parallel processing.")
    parser.add_argument('--pid', type=int, help='Process ID to monitor (default: monitor all EXO processes).')
    parser.add_argument('--interval', type=int, default=5, help='Monitoring interval in seconds.')
    parser.add_argument('--output_dir', type=str, default='./benchmarks/results', help='Directory to save monitoring results.')
    parser.add_argument('--duration', type=int, default=0, help='Duration to monitor in seconds (0 = until manually stopped).')
    
    args = parser.parse_args()
    
    print(f"Starting EXO performance monitoring")
    print(f"Interval: {args.interval} seconds")
    
    if args.pid:
        print(f"Monitoring specific process ID: {args.pid}")
    else:
        print("Monitoring all EXO processes")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.output_dir, f"exo_monitoring_{timestamp}.json")
    
    # Initialize metrics collection
    metrics = []
    start_time = time.time()
    
    try:
        while True:
            current_time = time.time()
            elapsed = current_time - start_time
            
            # Check if monitoring duration has been reached
            if args.duration > 0 and elapsed > args.duration:
                print(f"Monitoring duration of {args.duration} seconds reached. Stopping.")
                break
                
            # Get process metrics (placeholder for actual implementation)
            # In a real implementation, you would use ps, top, or a library like psutil
            # to get CPU usage, memory consumption, etc.
            metric = {
                "timestamp": current_time,
                "elapsed_seconds": elapsed,
                "cpu_percent": 0,  # Placeholder
                "memory_mb": 0,    # Placeholder
                "processes": 0     # Placeholder
            }
            
            metrics.append(metric)
            print(f"Collected metrics at {elapsed:.2f} seconds")
            
            # Save current metrics to file
            with open(output_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            time.sleep(args.interval)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")
    
    print(f"Monitoring data saved to: {output_file}")

if __name__ == "__main__":
    main()
