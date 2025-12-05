"""
Minimal example: File-based checkpointing vs MemorySaver
This demonstrates how state persists across program restarts.
"""

import json
import os
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict

# Simple state
class CounterState(TypedDict):
    count: int
    step: str

# Simple node that increments counter
def increment(state: CounterState):
    new_count = state.get("count", 0) + 1
    print(f"Step {new_count}: Incrementing counter to {new_count}")
    return {"count": new_count, "step": f"step_{new_count}"}

# Build graph
graph = StateGraph(CounterState)
graph.add_node("increment", increment)
graph.add_edge(START, "increment")
graph.add_edge("increment", END)

# OPTION 1: MemorySaver (lost on restart)
print("=== Testing MemorySaver (in-memory) ===")
memory_checkpointer = MemorySaver()
memory_graph = graph.compile(checkpointer=memory_checkpointer)

config = {"configurable": {"thread_id": "test-1"}}
result1 = memory_graph.invoke({"count": 0}, config=config)
print(f"Result 1: {result1}")

# OPTION 2: Manual file-based persistence
print("\n=== Testing File-based persistence (manual save/load) ===")
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def save_state_to_file(thread_id: str, state_data: dict):
    """Save state to a JSON file"""
    file_path = os.path.join(CHECKPOINT_DIR, f"{thread_id}.json")
    with open(file_path, "w") as f:
        json.dump(state_data, f, indent=2)
    print(f"State saved to {file_path}")

def load_state_from_file(thread_id: str) -> dict:
    """Load state from a JSON file"""
    file_path = os.path.join(CHECKPOINT_DIR, f"{thread_id}.json")
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)
    return None

# Use MemorySaver but manually save/load
file_checkpointer = MemorySaver()
file_graph = graph.compile(checkpointer=file_checkpointer)

config2 = {"configurable": {"thread_id": "test-2"}}
result2 = file_graph.invoke({"count": 0}, config=config2)
print(f"Result 2: {result2}")

# Manually save state
current_state = file_graph.get_state(config2)
save_state_to_file("test-2", {
    "values": current_state.values,
    "next": list(current_state.next) if current_state.next else [],
    "config": config2
})

# Simulate restart: Create new graph and load state
print("\n=== Simulating server restart ===")
print("(In real scenario, server would restart here)")
restored_data = load_state_from_file("test-2")
if restored_data:
    print(f"Restored state: {restored_data['values']}")
    print(f"Can resume from: {restored_data['next']}")
    
    # Resume workflow with restored state
    new_graph = graph.compile(checkpointer=MemorySaver())
    # Resume from where we left off
    if restored_data['next']:
        print(f"\nResuming workflow from checkpoint...")
        # In production, you'd use Command(resume={...}) here
        # For now, just show we can recover the state

# Add this after your existing code to demonstrate with ResumeState
print("\n" + "="*60)
print("=== Demonstrating with ResumeState (your actual workflow) ===")
print("="*60)

# Import your actual workflow components
from lesson11_parallel_nodes import ResumeState
from helpers import load_resume_data, load_job_description

# Show what state looks like when paused at human_in_loop
# (This simulates what would be saved)
example_resume_state = {
    "resume_data": load_resume_data(),
    "job_description": load_job_description(),
    "ats_score": 0.75,
    "skill_gap_score": 0.60,
    "relevant_experience": "5 years of Python development...",
    "extracted_skills": ["Python", "React", "Node.js"],
    "errors": [],
    "analysis_times": {"ats": 0.5, "skill_gap": 0.3, "experience": 1.2},
    "user_decision": "pending"
}

print("\nWhat gets saved when workflow pauses at 'human_in_loop':")
print(f"- Current node: human_in_loop")
print(f"- Full ResumeState values:")
for key, value in example_resume_state.items():
    if key == "resume_data":
        print(f"  - {key}: <resume data dict with {len(value)} sections>")
    elif key == "job_description":
        print(f"  - {key}: <job description text ({len(value)} chars)>")
    elif isinstance(value, dict):
        print(f"  - {key}: {value}")
    elif isinstance(value, list):
        print(f"  - {key}: {value}")
    else:
        print(f"  - {key}: {value}")

# Show what the checkpoint file would contain
checkpoint_data = {
    "values": example_resume_state,  # <-- ALL state data is here!
    "next": ["human_in_loop"],  # <-- This tells us where to resume
    "config": {"configurable": {"thread_id": "resume-123"}}
}

print(f"\nCheckpoint file structure:")
print(f"  - values: Contains ALL ResumeState fields (resume_data, scores, etc.)")
print(f"  - next: ['human_in_loop'] (where to resume)")
print(f"  - config: thread_id for this workflow session")

# Save it to see the actual structure
save_state_to_file("resume-example", checkpoint_data)
print(f"\nSaved example checkpoint to checkpoints/resume-example.json")
print("Check that file to see ALL the state data that gets saved!")

# ============================================================================
# PART 3: Checkpoint Versioning, Rollback, and Cleanup
# ============================================================================

import time
from datetime import datetime, timedelta
from pathlib import Path

class CheckpointManager:
    """
    Manages checkpoint versioning with rollback and cleanup capabilities.
    Only saves checkpoints at important nodes to save storage space.
    """
    
    def __init__(self, checkpoint_dir: str = "checkpoints", max_checkpoints: int = 15, ttl_hours: int = 24):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.ttl_hours = ttl_hours
    
    def save_checkpoint(self, thread_id: str, state_data: dict, node_name: str = None):
        """
        Save a checkpoint with versioning.
        Automatically cleans up old checkpoints beyond max_checkpoints limit.
        """
        timestamp = datetime.now().isoformat()
        checkpoint_id = f"{thread_id}_{int(time.time())}"
        
        checkpoint_data = {
            "checkpoint_id": checkpoint_id,
            "thread_id": thread_id,
            "timestamp": timestamp,
            "node_name": node_name,
            "values": state_data.get("values", {}),
            "next": list(state_data.get("next", [])),
            "config": state_data.get("config", {})
        }
        
        # Save checkpoint
        file_path = self.checkpoint_dir / f"{checkpoint_id}.json"
        with open(file_path, "w") as f:
            json.dump(checkpoint_data, f, indent=2)
        
        print(f"✓ Checkpoint saved: {checkpoint_id} (node: {node_name or 'unknown'})")
        
        # Clean up old checkpoints for this thread_id
        self._cleanup_old_checkpoints(thread_id)
        
        return checkpoint_id
    
    def list_checkpoints(self, thread_id: str) -> list:
        """List all checkpoints for a given thread_id, sorted by timestamp (newest first)"""
        checkpoints = []
        for file_path in self.checkpoint_dir.glob(f"{thread_id}_*.json"):
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    checkpoints.append({
                        "checkpoint_id": data["checkpoint_id"],
                        "timestamp": data["timestamp"],
                        "node_name": data.get("node_name"),
                        "file_path": str(file_path)
                    })
            except Exception as e:
                print(f"Error reading checkpoint {file_path}: {e}")
        
        # Sort by timestamp (newest first)
        checkpoints.sort(key=lambda x: x["timestamp"], reverse=True)
        return checkpoints
    
    def get_checkpoint(self, checkpoint_id: str) -> dict:
        """Load a specific checkpoint by ID"""
        file_path = self.checkpoint_dir / f"{checkpoint_id}.json"
        if not file_path.exists():
            return None
        
        with open(file_path, "r") as f:
            return json.load(f)
    
    def rollback_to_checkpoint(self, checkpoint_id: str) -> dict:
        """
        Rollback to a specific checkpoint.
        Returns the checkpoint data that can be used to restore workflow state.
        """
        checkpoint = self.get_checkpoint(checkpoint_id)
        if not checkpoint:
            raise ValueError(f"Checkpoint {checkpoint_id} not found")
        
        print(f"✓ Rolled back to checkpoint: {checkpoint_id}")
        print(f"  Timestamp: {checkpoint['timestamp']}")
        print(f"  Node: {checkpoint.get('node_name', 'unknown')}")
        
        return checkpoint
    
    def _cleanup_old_checkpoints(self, thread_id: str):
        """Remove old checkpoints beyond max_checkpoints limit"""
        checkpoints = self.list_checkpoints(thread_id)
        
        # Keep only the most recent max_checkpoints
        if len(checkpoints) > self.max_checkpoints:
            checkpoints_to_delete = checkpoints[self.max_checkpoints:]
            for checkpoint in checkpoints_to_delete:
                try:
                    Path(checkpoint["file_path"]).unlink()
                    print(f"  Deleted old checkpoint: {checkpoint['checkpoint_id']}")
                except Exception as e:
                    print(f"  Error deleting checkpoint {checkpoint['checkpoint_id']}: {e}")
    
    def cleanup_expired_checkpoints(self):
        """Remove checkpoints older than TTL"""
        cutoff_time = datetime.now() - timedelta(hours=self.ttl_hours)
        deleted_count = 0
        
        for file_path in self.checkpoint_dir.glob("*.json"):
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    checkpoint_time = datetime.fromisoformat(data["timestamp"])
                    
                    if checkpoint_time < cutoff_time:
                        file_path.unlink()
                        deleted_count += 1
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        if deleted_count > 0:
            print(f"✓ Cleaned up {deleted_count} expired checkpoints (older than {self.ttl_hours} hours)")
        return deleted_count


# ============================================================================
# Demonstration: Checkpoint Versioning with Rollback
# ============================================================================

print("\n" + "="*60)
print("=== Part 3: Checkpoint Versioning & Rollback ===")
print("="*60)

# Create checkpoint manager
checkpoint_mgr = CheckpointManager(max_checkpoints=15, ttl_hours=24)

# Simulate a workflow with multiple checkpoints
thread_id = "resume-optimization-123"

print(f"\nSimulating workflow: {thread_id}")
print("Saving checkpoints at important nodes...\n")

# Checkpoint 1: After parse_resume
checkpoint1_data = {
    "values": {"resume_data": {"name": "John Doe"}, "job_description": "Python developer"},
    "next": ["check_ats_score", "check_skill_gap_score"],
    "config": {"configurable": {"thread_id": thread_id}}
}
checkpoint1_id = checkpoint_mgr.save_checkpoint(thread_id, checkpoint1_data, "parse_resume")

# Simulate some time passing
time.sleep(0.1)

# Checkpoint 2: After aggregate_analyses (before human_in_loop)
checkpoint2_data = {
    "values": {
        "resume_data": {"name": "John Doe"},
        "job_description": "Python developer",
        "ats_score": 0.75,
        "skill_gap_score": 0.60,
        "relevant_experience": "5 years of Python..."
    },
    "next": ["human_in_loop"],
    "config": {"configurable": {"thread_id": thread_id}}
}
checkpoint2_id = checkpoint_mgr.save_checkpoint(thread_id, checkpoint2_data, "aggregate_analyses")

time.sleep(0.1)

# Checkpoint 3: After apply_suggestions
checkpoint3_data = {
    "values": {
        "resume_data": {"name": "John Doe", "optimized": True},
        "job_description": "Python developer",
        "ats_score": 0.85,  # Improved!
        "skill_gap_score": 0.70,  # Improved!
        "user_decision": "1"
    },
    "next": ["parse_resume"],  # Loop back
    "config": {"configurable": {"thread_id": thread_id}}
}
checkpoint3_id = checkpoint_mgr.save_checkpoint(thread_id, checkpoint3_data, "apply_suggestions")

# List all checkpoints
print(f"\n--- All checkpoints for {thread_id} ---")
checkpoints = checkpoint_mgr.list_checkpoints(thread_id)
for i, cp in enumerate(checkpoints, 1):
    print(f"{i}. {cp['checkpoint_id']}")
    print(f"   Node: {cp['node_name']}")
    print(f"   Time: {cp['timestamp']}")

# Demonstrate rollback
print(f"\n--- Rollback Demonstration ---")
print("Current state (checkpoint 3):")
current = checkpoint_mgr.get_checkpoint(checkpoint3_id)
print(f"  ATS Score: {current['values'].get('ats_score', 'N/A')}")

print("\nRolling back to checkpoint 2 (before optimizations)...")
rolled_back = checkpoint_mgr.rollback_to_checkpoint(checkpoint2_id)
print(f"  ATS Score: {rolled_back['values'].get('ats_score', 'N/A')}")
print(f"  Can resume from node: {rolled_back['next']}")

# Show cleanup
print(f"\n--- Cleanup Demonstration ---")
print("Cleaning up expired checkpoints...")
deleted = checkpoint_mgr.cleanup_expired_checkpoints()

print(f"\n✓ Checkpoint versioning system ready!")
print(f"  - Max checkpoints per session: {checkpoint_mgr.max_checkpoints}")
print(f"  - TTL: {checkpoint_mgr.ttl_hours} hours")
print(f"  - Rollback: Restore to any previous checkpoint")
print(f"  - Auto-cleanup: Old checkpoints removed automatically")