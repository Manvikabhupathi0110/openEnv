#!/usr/bin/env bash
set -e

echo "=== OpenEnv Electrician Scheduling - Pre-Submission Validator ==="
echo ""

PASS=0
FAIL=0

check() {
    local name="$1"
    local result="$2"
    if [ "$result" = "0" ]; then
        echo "✓ $name"
        PASS=$((PASS+1))
    else
        echo "✗ $name"
        FAIL=$((FAIL+1))
    fi
}

# 1. Check openenv.yaml exists
[ -f openenv.yaml ] && check "openenv.yaml exists" 0 || check "openenv.yaml exists" 1

# 2. Check required Python files
[ -f openenv_electrician/environment.py ] && check "environment.py exists" 0 || check "environment.py exists" 1
[ -f openenv_electrician/models.py ] && check "models.py exists" 0 || check "models.py exists" 1
[ -f openenv_electrician/tasks.py ] && check "tasks.py exists" 0 || check "tasks.py exists" 1
[ -f server/app.py ] && check "server/app.py exists" 0 || check "server/app.py exists" 1

# 3. Check inference.py
[ -f inference.py ] && check "inference.py exists" 0 || check "inference.py exists" 1

# 4. Check Dockerfile
[ -f Dockerfile ] && check "Dockerfile exists" 0 || check "Dockerfile exists" 1

# 5. Check README
[ -f README.md ] && check "README.md exists" 0 || check "README.md exists" 1

# 6. Python import test
python -c "from openenv_electrician.environment import ElectricianSchedulingEnv; env = ElectricianSchedulingEnv(); print('Import OK')" 2>/dev/null \
    && check "Python import" 0 || check "Python import" 1

# 7. Reset test
python -c "
from openenv_electrician.environment import ElectricianSchedulingEnv
env = ElectricianSchedulingEnv()
obs = env.reset(task_name='easy')
assert obs is not None
assert hasattr(obs, 'done')
print('reset() OK')
" 2>/dev/null && check "reset() works" 0 || check "reset() works" 1

# 8. Step test
python -c "
from openenv_electrician.environment import ElectricianSchedulingEnv
env = ElectricianSchedulingEnv()
env.reset(task_name='easy')
obs = env.step({'type': 'list_tickets'})
assert obs is not None
print('step() OK')
" 2>/dev/null && check "step() works" 0 || check "step() works" 1

# 9. State test
python -c "
from openenv_electrician.environment import ElectricianSchedulingEnv
env = ElectricianSchedulingEnv()
env.reset(task_name='easy')
s = env.state
assert s is not None
print('state() OK')
" 2>/dev/null && check "state works" 0 || check "state works" 1

# 10. All 3 tasks grader test
python -c "
from openenv_electrician.tasks import grade_easy, grade_medium, grade_hard
state = {'tickets': [], 'electricians': [], 'appointments': []}
for grader in [grade_easy, grade_medium, grade_hard]:
    score = grader(state, [])
    assert 0.0 < score < 1.0, f'Score out of range: {score}'
print('Graders OK')
" 2>/dev/null && check "Graders return [0,1] scores" 0 || check "Graders return [0,1] scores" 1

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ $FAIL -eq 0 ] && echo "✓ All checks passed!" || echo "✗ Some checks failed"
exit $FAIL
