#!/bin/bash
# Start both server and webui

# Start FastAPI server in background
python server.py &
SERVER_PID=$!

# Forward docker stop's SIGTERM to both processes so GPU memory is freed promptly
trap 'kill $SERVER_PID 2>/dev/null; kill $WEBUI_PID 2>/dev/null; exit 0' TERM INT

# Wait for server to be ready
echo "Waiting for server..."
for i in {1..30}; do
    if curl -s http://localhost:8765/health > /dev/null; then
        echo "Server ready"
        break
    fi
    sleep 1
done

# Start Flask webui in background so the trap above can catch signals
(cd webui && python app.py) &
WEBUI_PID=$!

wait -n $SERVER_PID $WEBUI_PID
kill $SERVER_PID $WEBUI_PID 2>/dev/null
