import subprocess
import os
import sys

def run_bruteforce(config_path, tm_index):
    """Run bruteforce analysis for a specific traffic matrix and save results"""
    output_file = f"bruteforce_results_tm{tm_index}.txt"
    
    # Command to run
    cmd = f"python simple_bruteforce.py --config {config_path} --tm-index {tm_index}"
    
    # Run the command and capture output
    print(f"Running bruteforce for TM {tm_index}...")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    # Save output to file
    with open(output_file, "w") as f:
        f.write(f"Bruteforce Results for Traffic Matrix {tm_index}\n")
        f.write("="*80 + "\n\n")
        f.write(result.stdout)
        if result.stderr:
            f.write("\nErrors:\n")
            f.write(result.stderr)
    
    print(f"Results saved to {output_file}")
    
    # Also print the best solution part
    lines = result.stdout.split('\n')
    best_solution_found = False
    for i, line in enumerate(lines):
        if "Best solution for" in line:
            best_solution_found = True
            result_lines = lines[i:i+6]
            print("\n".join(result_lines))
            break
    
    if not best_solution_found:
        print("Could not find best solution in output.")

def main():
    if len(sys.argv) < 3:
        print("Usage: python run_bruteforce_analysis.py <config_path> <tm_index>")
        return
    
    config_path = sys.argv[1]
    tm_index = int(sys.argv[2])
    run_bruteforce(config_path, tm_index)

if __name__ == "__main__":
    main()
