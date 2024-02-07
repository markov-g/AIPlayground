docker-compose -f docker-compose.yml -f docker-compose-ollama.yml up # -f docker-compose-jupyter.yml -f docker-compose-pgadmin.yml up
sleep 10
open http://localhost:7800
