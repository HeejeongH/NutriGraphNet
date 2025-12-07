"""
전체 파이프라인 검증 스크립트
"""
import sys
import os
from pathlib import Path

print("=" * 70)
print("🔍 NutriGraphNet 파이프라인 검증")
print("=" * 70)

errors = []
warnings = []

# 1. 필수 파일 존재 확인
print("\n1️⃣ 필수 파일 확인:")
required_files = [
    'train_v2.py',
    'run_health_experiments.sh',
    'src/run_health_aware_experiments.py',
    'src/compare_health_results.py',
    'src/evaluation_metrics.py',
    'src/health_score_calculator.py',
    'data/graph_builder.py',
    'data/processed_data/processed_data_GNN_fixed.pkl'
]

for file in required_files:
    if Path(file).exists():
        print(f"   ✅ {file}")
    else:
        print(f"   ❌ {file} - NOT FOUND")
        errors.append(f"Missing file: {file}")

# 2. train_v2.py 인자 확인
print("\n2️⃣ train_v2.py 인자 확인:")
sys.path.append('src')

try:
    # Try UTF-8 first, fallback to system encoding
    try:
        with open('train_v2.py', 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        # Fallback for systems with different default encoding
        import locale
        system_encoding = locale.getpreferredencoding()
        with open('train_v2.py', 'r', encoding=system_encoding) as f:
            content = f.read()
        
    required_args = [
        '--data_path',
        '--model',
        '--epochs',
        '--hidden_channels',
        '--out_channels',
        '--loss',
        '--result_file'
    ]
    
    for arg in required_args:
        if arg in content:
            print(f"   ✅ {arg}")
        else:
            print(f"   ❌ {arg} - NOT FOUND")
            errors.append(f"Missing argument in train_v2.py: {arg}")
            
except Exception as e:
    print(f"   ❌ Error reading train_v2.py: {e}")
    errors.append(str(e))

# 3. run_health_experiments.sh 명령어 확인
print("\n3️⃣ run_health_experiments.sh 명령어 확인:")
try:
    # Try UTF-8 first, fallback to system encoding
    try:
        with open('run_health_experiments.sh', 'r', encoding='utf-8') as f:
            script_content = f.read()
    except UnicodeDecodeError:
        import locale
        system_encoding = locale.getpreferredencoding()
        with open('run_health_experiments.sh', 'r', encoding=system_encoding) as f:
            script_content = f.read()
    
    # --result_file이 모든 실험에 있는지 확인
    if script_content.count('--result_file') >= 6:
        print(f"   ✅ --result_file 옵션 (6개 이상 발견)")
    else:
        count = script_content.count('--result_file')
        print(f"   ⚠️  --result_file 옵션 ({count}개 발견, 6개 필요)")
        warnings.append(f"--result_file count mismatch: {count}/6")
    
    # 결과 비교 스크립트 경로 확인
    if 'python src/compare_health_results.py' in script_content:
        print(f"   ✅ 결과 비교 스크립트 경로 (src/)")
    elif 'compare_health_results.py' in script_content and 'src/' not in script_content:
        print(f"   ❌ 결과 비교 스크립트 경로 오류 (src/ 없음)")
        errors.append("compare_health_results.py path should be src/")
    else:
        print(f"   ⚠️  결과 비교 스크립트를 찾을 수 없음")
        warnings.append("compare_health_results.py not found in script")
        
except Exception as e:
    print(f"   ❌ Error reading run_health_experiments.sh: {e}")
    errors.append(str(e))

# 4. Python import 테스트
print("\n4️⃣ Python 모듈 import 테스트:")
try:
    sys.path.insert(0, 'src')
    
    # torch 확인
    try:
        import torch
        print(f"   ✅ torch (PyTorch)")
    except ImportError:
        print(f"   ⚠️  torch not installed (PyTorch required for training)")
        warnings.append("PyTorch not installed - required for actual training")
    
    modules_to_test = [
        ('evaluation_metrics', 'compute_comprehensive_metrics'),
        ('health_score_calculator', 'PersonalizedHealthScoreCalculator'),
    ]
    
    for module_name, class_name in modules_to_test:
        try:
            module = __import__(module_name)
            if hasattr(module, class_name):
                print(f"   ✅ {module_name}.{class_name}")
            else:
                print(f"   ⚠️  {module_name} imported but {class_name} not found")
                warnings.append(f"{class_name} not found in {module_name}")
        except ImportError as e:
            if 'torch' in str(e):
                print(f"   ⚠️  {module_name} (requires PyTorch)")
                warnings.append(f"{module_name} requires PyTorch")
            else:
                print(f"   ❌ {module_name} - {str(e)[:50]}")
                errors.append(f"Import error: {module_name}")
        except Exception as e:
            print(f"   ❌ {module_name} - {str(e)[:50]}")
            errors.append(f"Import error: {module_name}")
            
except Exception as e:
    print(f"   ❌ Import test failed: {e}")
    errors.append(str(e))

# 5. 결과 디렉토리 생성 테스트
print("\n5️⃣ 결과 디렉토리 생성 테스트:")
test_dir = Path('results/health_experiments')
try:
    test_dir.mkdir(parents=True, exist_ok=True)
    if test_dir.exists():
        print(f"   ✅ results/health_experiments/ 생성 가능")
    else:
        print(f"   ❌ 디렉토리 생성 실패")
        errors.append("Cannot create results directory")
except Exception as e:
    print(f"   ❌ Error: {e}")
    errors.append(str(e))

# 최종 결과
print("\n" + "=" * 70)
print("📊 검증 결과")
print("=" * 70)

if len(errors) == 0:
    print("\n✅ 핵심 검사 통과! 실험 실행 준비 완료")
    
    if len(warnings) > 0:
        print(f"\n⚠️  {len(warnings)}개 경고 (무시 가능):")
        for warn in warnings:
            print(f"   • {warn}")
        print("\n   ※ torch 관련 경고는 로컬 Mac 실행 시 정상 동작합니다")
    
    print("\n🚀 실행 방법:")
    print("   bash run_health_experiments.sh")
    sys.exit(0)
else:
    print(f"\n❌ {len(errors)}개 치명적 오류 발견:")
    for err in errors:
        print(f"   • {err}")
    
    if len(warnings) > 0:
        print(f"\n⚠️  {len(warnings)}개 경고:")
        for warn in warnings:
            print(f"   • {warn}")
    
    print("\n🔧 문제를 수정한 후 다시 실행하세요")
    sys.exit(1)

