from locust import HttpUser, task, between


class AgentUser(HttpUser):
    # Wait time between tasks (in seconds)
    wait_time = between(1, 3)

    @task
    def test_agent_respond(self):
        # Define the payload for the POST request
        payload = {"message": "Hello, how can I update my identity?"}

        # Send a POST request to the /agent/respond endpoint
        with self.client.post(
            "/agent/respond", json=payload, catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(
                    f"Failed with status code {response.status_code}: {response.text}"
                )
