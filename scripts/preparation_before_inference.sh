
export DATASET=webarena
# 1. export commands
source export_commands.sh
# 2. generate test data
python -m scripts.generate_test_data
# 3. prepare 
bash prepare.sh