import os
import subprocess


def install():
    save_dir = 'vendor'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    hifiasm_dir_name = f'hifiasm-0.23.0'
    if os.path.isfile(os.path.join(save_dir, hifiasm_dir_name, 'hifiasm')):
        print(f'\nFound hifiasm! Skipping installation...\n')
    else:
        # Install hifiasm
        print(f'\nInstalling hifisam...')
        subprocess.run(f'git clone https://github.com/chhylp123/hifiasm.git --branch 0.23.0 --single-branch {hifiasm_dir_name}', shell=True, cwd=save_dir)
        hifiasm_dir = os.path.join(save_dir, hifiasm_dir_name)
        subprocess.run(f'make', shell=True, cwd=hifiasm_dir)
        # print(f'Install hifiasm version: ', end='')
        # subprocess.run(f'./hifiasm --version', shell=True, cwd=hifiasm_dir)
        # print()

    hifiasm_dir_name = f'hifiasm-0.18.8'
    if os.path.isfile(os.path.join(save_dir, hifiasm_dir_name, 'hifiasm')):
        print(f'\nFound hifiasm! Skipping installation...\n')
    else:
        # Install hifiasm
        print(f'\nInstalling hifisam...')
        subprocess.run(f'git clone https://github.com/chhylp123/hifiasm.git --branch 0.18.8 --single-branch {hifiasm_dir_name}', shell=True, cwd=save_dir)
        hifiasm_dir = os.path.join(save_dir, hifiasm_dir_name)
        subprocess.run(f'make', shell=True, cwd=hifiasm_dir)
        # print(f'Install hifiasm version: ', end='')
        # subprocess.run(f'./hifiasm --version', shell=True, cwd=hifiasm_dir)
        # print()

    raven_dir_name = f'heron-raven_ftomas-dev'
    if os.path.isfile(os.path.join(save_dir, raven_dir_name, 'build', 'bin', 'raven')):
        print(f'\nFound raven! Skipping installation...\n')
    else:
        # Install Raven
        print(f'\nInstalling Raven...')
        subprocess.run(f'git clone https://github.com/tihomirkonosic/heron-raven.git --branch ftomas/dev --single-branch {raven_dir_name}', shell=True, cwd=save_dir)
        raven_dir = os.path.join(save_dir, raven_dir_name)
        raven_build = os.path.join(raven_dir, 'build')
        os.mkdir(raven_build)
        subprocess.run(f'cmake -DCMAKE_BUILD_TYPE=Release .. && make', shell=True, cwd=raven_build)
        # print(f'Install Raven version: ', end='')
        # subprocess.run(f'./build/bin/raven --version', shell=True, cwd=raven_dir)
        # print()

    pbsim_dir_name = f'pbsim3'
    if os.path.isfile(os.path.join(save_dir, pbsim_dir_name, 'src', 'pbsim')):
        print(f'\nFound PBSIM3! Skipping installation...\n')
    else:
        # Install PBSIM3
        print(f'\nInstalling PBSIM3...')
        subprocess.run(f'git clone https://github.com/yukiteruono/pbsim3.git {pbsim_dir_name}', shell=True, cwd=save_dir)
        pbsim_dir = os.path.join(save_dir, pbsim_dir_name)
        subprocess.run(f'./configure; make; make install', shell=True, cwd=pbsim_dir)
        # print(f'Install PBSIM version: ', end='')
        # subprocess.run(f'./src/pbsim --version', shell=True, cwd=pbsim_dir)
        # print()

    tool_name = f'yak'
    if os.path.isfile(os.path.join(save_dir, tool_name, 'yak')):
        print(f'\nFound {tool_name}! Skipping installation...\n')
    else:
        # Install yak
        print(f'\nInstalling {tool_name}...')
        subprocess.run(f'git clone https://github.com/lh3/yak.git', shell=True, cwd=save_dir)
        tool_dir = os.path.join(save_dir, tool_name)
        subprocess.run(f'make', shell=True, cwd=tool_dir)

    tool_name = f'RAFT'
    if os.path.isfile(os.path.join(save_dir, tool_name, 'raft')):
        print(f'\nFound {tool_name}! Skipping installation...\n')
    else:
        # Install RAFT
        print(f'\nInstalling {tool_name}...')
        subprocess.run(f'git clone https://github.com/at-cg/RAFT.git', shell=True, cwd=save_dir)
        tool_dir = os.path.join(save_dir, tool_name)
        subprocess.run(f'make', shell=True, cwd=tool_dir)


    tool_name = f'Badread'
    if os.path.isfile(os.path.join(save_dir, tool_name, 'Badread')):
        print(f'\nFound {tool_name}! Skipping installation...\n')
    else:
        # Install Badreads
        print(f'\nInstalling {tool_name}...')
        subprocess.run(f'git clone https://github.com/rrwick/Badread.git', shell=True, cwd=save_dir)
        tool_dir = os.path.join(save_dir, tool_name)
        subprocess.run(f'pip install ./Badread', shell=True, cwd=save_dir)

if __name__ == '__main__':
    install()

