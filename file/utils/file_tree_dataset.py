"""File tree dataset implementation for model exploration strategy research.

This module provides the core data structures and functionality for creating
and managing file trees that can be explored by models.
"""

import json
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field


@dataclass
class FileTreeNode:
    """Represents a node in the file tree with dual file/folder properties.

    Every node stores a filename-like name, optional content, and children. At
    runtime, nodes with children are treated as folders and leaf nodes are
    treated as files.
    """
    name: str
    content: str = ""
    children: List['FileTreeNode'] = field(default_factory=list)
    depth: int = 0
    parent: Optional['FileTreeNode'] = None
    
    @property
    def type(self) -> str:
        """Return the display type implied by the node's children."""
        return "folder" if self.children else "file"
    
    def is_file(self) -> bool:
        """Check if this node should be treated as a file (no children)."""
        return len(self.children) == 0
    
    def is_folder(self) -> bool:
        """Check if this node should be treated as a folder (has children)."""
        return len(self.children) > 0
    
    def validate(self) -> bool:
        """Validate that node has proper structure."""
        # All nodes except root should have file extensions
        if self.name != "root" and "." not in self.name:
            return False
        
        # Recursively validate children
        for child in self.children:
            if not child.validate():
                return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary representation."""
        result = {
            "name": self.name,
            "content": self.content,
            "depth": self.depth
        }
        if self.children:
            result["children"] = [child.to_dict() for child in self.children]
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], depth: int = 0, parent: Optional['FileTreeNode'] = None) -> 'FileTreeNode':
        """Create node from dictionary representation with proper depth tracking."""
        content = data.get("content", "")
        if not isinstance(content, str):
            import json
            content = json.dumps(content, ensure_ascii=False, indent=2)

        node = cls(
            name=data.get("name", "unnamed"),
            content=content,
            depth=depth,
            parent=parent
        )
        if "children" in data:
            node.children = [cls.from_dict(child, depth + 1, node) for child in data["children"]]
        return node


class FileTree:
    """Manages file tree structure with save/load and access functionality."""
    
    def __init__(self, root: FileTreeNode):
        """Initialize with root node."""
        self.root = root
        self.current_path = []  # Track current directory path
    
    def save(self, filepath: str) -> None:
        """Save file tree to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.root.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, filepath: str) -> 'FileTree':
        """Load file tree from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        root = FileTreeNode.from_dict(data, depth=0, parent=None)
        return cls(root)
    
    def get_current_node(self) -> FileTreeNode:
        """Get the current directory node based on current path."""
        node = self.root
        for folder_name in self.current_path:
            for child in node.children:
                if child.name == folder_name:
                    node = child
                    break
        return node
    
    def get_path_to_node(self, target_node: FileTreeNode) -> List[str]:
        """Get the path from root to a specific node.
        
        Args:
            target_node: The node to find path to
            
        Returns:
            List of node names from root to target (excluding root)
        """
        path = []
        current = target_node
        while current.parent is not None:
            path.append(current.name)
            current = current.parent
        path.reverse()
        return path
    
    def list_directory(self) -> List[Dict[str, str]]:
        """List contents of current directory.
        
        Returns:
            List of dicts with 'name' and 'type' for each item
        """
        current = self.get_current_node()
        items = []
        for child in current.children:
            items.append({
                "name": child.name,
                "type": child.type
            })
        return items
    
    def view_file(self, filename: str) -> Optional[str]:
        """View content of a file in current directory.
        
        Args:
            filename: Name of the file to view (must have extension)
            
        Returns:
            File content if found and is a valid file, None otherwise
        """
        current = self.get_current_node()
        for child in current.children:
            if child.name == filename and child.is_file():
                return child.content
        return None
    
    def enter_folder(self, path: str) -> bool:
        """Enter a folder using only the three restricted path formats.
        
        Args:
            path: Must be one of:
                - Simple folder name (e.g., "documents") - no "/" allowed
                - ".." to go to parent directory
                - Absolute path starting with "/" (e.g., "/root/folder1/folder2")
            
        Returns:
            True if successful, False if folder not found or invalid format
        """
        # Strict path validation - reject unsupported formats
        if "/" in path and not path.startswith("/"):
            # Reject relative paths with "/" like "folder1/folder2" or "./folder"
            return False
        elif path.startswith("./"):
            # Reject paths with current directory prefix
            return False
        elif ".." in path and path != "..":
            # Reject paths containing ".." unless it's exactly ".."
            return False
        
        if path.startswith("/"):
            # Handle absolute path
            return self._enter_full_path(path)
        elif path == "..":
            # Handle parent directory only
            if self.current_path:
                self.current_path.pop()
                return True
            else:
                # Already at root
                return False
        else:
            # Handle simple folder name only
            return self._enter_simple_folder(path)
    
    def _enter_simple_folder(self, foldername: str) -> bool:
        """Enter a single subfolder in current directory by simple name only.
        
        This method only handles simple folder names without any path separators.
        Complex relative paths are no longer supported.
        
        Args:
            foldername: Simple folder name (no "/" allowed)
            
        Returns:
            True if folder found and entered, False otherwise
        """
        current = self.get_current_node()
        for child in current.children:
            if child.is_folder() and child.name == foldername:
                self.current_path.append(child.name)
                return True
        return False
    
    
    
    def _enter_full_path(self, full_path: str) -> bool:
        """Enter a folder using full path from root."""
        # Clean up the path
        path_parts = [p for p in full_path.split("/") if p]
        
        # If path is empty or just "/", go to root
        if not path_parts:
            self.current_path = []
            return True
        
        # Special handling for 'root' at the beginning
        if path_parts[0] == "root":
            path_parts = path_parts[1:]
        
        # Validate the full path exists and points to a folder
        node = self.root
        actual_path = []
        
        for part in path_parts:
            found = False
            for child in node.children:
                if child.is_folder() and child.name == part:
                    node = child
                    actual_path.append(child.name)
                    found = True
                    break
            if not found:
                return False
        
        # If we reach here, the path is valid
        self.current_path = actual_path
        return True
    
    
    def get_current_path_string(self) -> str:
        """Get current path as string."""
        if not self.current_path:
            return "/"
        return "/" + "/".join(self.current_path)
    
    def reset_to_root(self) -> None:
        """Reset current path to root directory."""
        self.current_path = []
    
    def get_node_by_path(self, path: List[str]) -> Optional[FileTreeNode]:
        """Get node by path list starting from root."""
        if not path or path[0] != "root":
            return None
        
        node = self.root
        for folder_name in path[1:]:  # Skip "root"
            found = False
            for child in node.children:
                if child.name == folder_name:
                    node = child
                    found = True
                    break
            if not found:
                return None
        return node
    
    def verify_path_to_target(self, path_string: str, target_filename: str) -> bool:
        """Verify that a path string leads to the target file in the tree.
        
        Args:
            path_string: The claimed path (e.g., "/folder1/folder2/target.txt")
            target_filename: The target filename to find
            
        Returns:
            True if the path is valid and leads to the target file
        """
        # Clean up the path string
        path_string = path_string.strip()
        
        # Remove leading slash if present
        if path_string.startswith("/"):
            path_string = path_string[1:]
        
        # Split into components
        path_parts = [p for p in path_string.split("/") if p]
        
        if not path_parts:
            return False
        
        # The last part should be the target filename
        if path_parts[-1] != target_filename:
            return False
        
        # Navigate through the tree following the path
        node = self.root
        
        # For each folder in the path (excluding the final file)
        for i, part in enumerate(path_parts[:-1]):
            found = False
            
            # Look for a child that matches this part
            for child in node.children:
                if child.is_folder() and child.name == part:
                    node = child
                    found = True
                    break
            
            if not found:
                return False
        
        # Check if the target file exists in the final folder
        for child in node.children:
            if child.is_file() and child.name == target_filename:
                return True
        
        return False
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FileTree':
        """Create FileTree from dictionary representation."""
        root = FileTreeNode.from_dict(data, depth=0, parent=None)
        return cls(root)
