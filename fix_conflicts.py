import os
import re
from pathlib import Path

def resolve_conflict(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    if '<<<<<<<' not in content:
        return False

    # Stashed changes (아래쪽, >>>>>>> 쪽) 버전 선택
    result = []
    in_ours = False
    in_theirs = False

    for line in content.split('\n'):
        if line.startswith('<<<<<<< '):
            in_ours = True
            in_theirs = False
        elif line.startswith('======='):
            in_ours = False
            in_theirs = True
        elif line.startswith('>>>>>>> '):
            in_ours = False
            in_theirs = False
        elif in_theirs:
            result.append(line)
        elif not in_ours and not in_theirs:
            result.append(line)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(result))
    return True

# results/analysis 하위 모든 JSON 처리
fixed = []
target_dir = Path('results/analysis')

if not target_dir.exists():
    print(f"ERROR: '{target_dir}' 폴더가 현재 디렉토리에 없습니다.")
    print(f"현재 디렉토리: {Path('.').resolve()}")
    print("이 스크립트를 NutriGraphNet 폴더 안에서 실행해 주세요.")
else:
    json_files = list(target_dir.rglob('*.json'))
    print(f"검색된 JSON 파일 수: {len(json_files)}")

    for p in json_files:
        if resolve_conflict(p):
            fixed.append(str(p))
            print(f'Fixed: {p}')

    print(f'\nTotal fixed: {len(fixed)} files')
    if fixed:
        print('\n수정된 파일 목록:')
        for f in fixed:
            print(f'  - {f}')
