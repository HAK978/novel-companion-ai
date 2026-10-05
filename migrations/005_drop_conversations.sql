-- The conversations table was created in 001 for server-side chat history and was never
-- written or read. Conversation history for follow-up questions is sent by the client with
-- each request, so nothing needs it.
DROP TABLE IF EXISTS conversations;
