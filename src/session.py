import time

class SessionManager:
    def __init__(self):
        # Stores conversation history: {session_id: [ {speaker, text, ts}, ... ]}
        self.sessions = {}

    def start_session(self, session_id: str) -> None:
        """Create a new empty session if it doesn’t exist."""
        if session_id not in self.sessions:
            self.sessions[session_id] = []

    def append_user_turn(self, session_id: str, text: str) -> None:
        """Append a user message to the session history."""
        if session_id not in self.sessions:
            self.start_session(session_id)
        self.sessions[session_id].append({
            "speaker": "user",
            "text": text,
            "ts": time.time()
        })

    def append_assistant_turn(self, session_id: str, text: str) -> None:
        """Append an assistant message to the session history."""
        if session_id not in self.sessions:
            self.start_session(session_id)
        self.sessions[session_id].append({
            "speaker": "assistant",
            "text": text,
            "ts": time.time()
        })

    def get_history(self, session_id: str):
        """Return the history of a session. Returns [] if not found."""
        return self.sessions.get(session_id, [])

    def clear_session(self, session_id: str) -> None:
        """Clear the history for a session."""
        if session_id in self.sessions:
            self.sessions[session_id] = []
