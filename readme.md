prerequsite: docker desktop

1. run the elastic search server
'''
docker run -d --name elasticsearch -p 9200:9200 -p 9300:9300 -e "discovery.type=single-node" -e "xpack.security.enabled=false" -e "xpack.security.http.ssl.enabled=false" elasticsearch:8.17.3
'''
2. setup a .env file

3. run the python fastAPI, using Dockerfile

4. call the api using this format, with docker ip
'''
curl -X POST "http://127.0.0.1:8000/agent/respond" \
-H "Content-Type: application/json" \
-d '{"message": "Hello, how can I help you?"}'
'''