# memory_manager.py
from collections import deque
from typing import Optional, Dict, Any, Tuple, Deque

class ChatMemory:
    """
    Manages short-term conversation memory, including JSON states.
    """
    def __init__(self, max_history_len: int = 5):
        # Store tuples of (user_query, assistant_response, json_before, json_after)
        self.history: Deque[Tuple[str, str, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]] = deque(maxlen=max_history_len)

    def add_turn(self, query: str, response: str, json_before: Optional[Dict[str, Any]], json_after: Optional[Dict[str, Any]]):
        """Adds a conversation turn to the memory."""
        # Ensure we're not storing duplicate states
        if self.history and json_after == self.history[-1][3]:
            return
            
        self.history.append((query, response, json_before, json_after))
        print(f"[Memory] Added turn. History size: {len(self.history)}")

    def get_last_turn(self) -> Optional[Tuple[str, str, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]]:
        """Returns the most recent conversation turn."""
        return self.history[-1] if self.history else None

    def get_previous_json_state(self) -> Optional[Dict[str, Any]]:
        """
        Returns the JSON state *before* the last modification.
        Useful for 'undo' or 'change it back'.
        """
        if len(self.history) < 1:
            return None
            
        # Find the most recent turn that had a JSON change
        for turn in reversed(self.history):
            if turn[2] is not None:  # json_before exists
                return turn[2]
        return None

    def get_recent_history_text(self, num_turns: int = 2) -> str:
        """Returns a formatted string of the last few turns for context."""
        if not self.history:
            return ""
            
        history_str = ""
        # Get the last 'num_turns' from the deque
        turns_to_show = list(self.history)[-num_turns:]
        for i, (q, r, _, _) in enumerate(turns_to_show):
            history_str += f"--- Turn {i+1} ---\nUser: {q}\nAssistant: {r}\n\n"
        return history_str.strip()