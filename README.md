## Tech Support Agent with LLM

This project implements a technical support agent that utilizes a large language model (LLM) to provide assistance with Apple product issues. The agent leverages a knowledge base of documents and retrieves relevant information to answer user queries.

### Features

* Accesses a knowledge base of Apple product documents (PDF, ePub, Word documents, TXT files)
* Retrieves relevant documents based on user queries
* Generates responses using a large language model (currently set to gpt-4)
* Provides up to 4 most likely solutions to user problems based on the knowledge base
* Maintains a conversation history to improve context for subsequent queries

### Requirements

* Python 3.7+
* OpenAI API Key (set via environment variable `OPENAI_API_KEY`)
* langchain library (`pip install langchain`)

### Usage

1. **Install langchain:**

   ```bash
   pip install langchain
   ```

2. **Set your OpenAI API key:**

   - Create an OpenAI account and obtain an API key.
   - Set the `OPENAI_API_KEY` environment variable with your key value.

3. **Prepare your knowledge base:**

   - Place your Apple product documents (PDF, ePub, Word, TXT) in a folder.

4. **Run the script:**

   ```bash
   python tech_support_agent.py
   ```

   The script will load your documents, configure the retriever and LLM, and start a chat session where you can ask questions about Apple products.

### Example

**Input:**

```
You: My iPhone screen is cracked. What should I do?
```

**Output:**

```
Tech Support Agent: Here are 4 closest suggestions for your cracked iPhone screen:

- Schedule an appointment at an Apple Store for a screen replacement.
- Visit an Apple Authorized Service Provider for screen repair.
- If your iPhone has AppleCare+, you might be eligible for a discounted screen replacement.
- Depending on the severity of the crack, you might be able to continue using your iPhone with a case.

Is there anything else I can help you with?
```

**Note:** This is a fictional example, and the actual number of suggestions and their content may vary depending on the knowledge base and the user's query.

### Additional Notes

* This is a basic example, and the agent can be further customized to improve the quality and accuracy of the responses.
* The provided documents should be relevant to Apple product troubleshooting and support. 
