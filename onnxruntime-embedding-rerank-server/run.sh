nohup uv run app/main.py > app.log 2>&1 &
PID1=$!

# 生成close.sh脚本，写入kill命令
echo "#!/bin/bash" > stop.sh
echo "kill -9 $PID1" >> stop.sh
