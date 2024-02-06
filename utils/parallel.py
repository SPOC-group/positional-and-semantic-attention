import subprocess
from concurrent.futures import ThreadPoolExecutor

def run_shell_command(command):
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {command}\n{e}")
        
def run_all_parallel(cmds):
    with ThreadPoolExecutor(max_workers=len(cmds)) as executor:
        executor.map(run_shell_command, cmds)
        executor.shutdown(wait=True, cancel_futures=True)