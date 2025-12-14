"""
Graph structure for symbol relationships.
Explicit graph representation of musical symbols and their connections.
"""
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .symbols import Symbol


class EdgeType(Enum):
    """Types of edges in the symbol graph."""
    STEM_NOTEHEAD = "stem_notehead"
    FLAG_STEM = "flag_stem"
    BEAM_STEM = "beam_stem"
    DOT_NOTEHEAD = "dot_notehead"
    ACCIDENTAL_NOTEHEAD = "accidental_notehead"
    BEAM_BEAM = "beam_beam"  # Multiple beams stacked
    UNKNOWN = "unknown"


class NodeType(Enum):
    """Types of nodes in the symbol graph."""
    NOTEHEAD = "notehead"
    STEM = "stem"
    FLAG = "flag"
    BEAM = "beam"
    DOT = "dot"
    ACCIDENTAL = "accidental"
    CLEF = "clef"
    BARLINE = "barline"
    REST = "rest"
    OTHER = "other"


@dataclass
class GraphNode:
    """
    Represents a node in the symbol graph.
    Each node corresponds to a detected symbol.
    """
    symbol: Symbol
    node_type: NodeType
    node_id: int = 0  # Unique identifier
    
    def __hash__(self):
        return hash(id(self.symbol))
    
    def __eq__(self, other):
        if not isinstance(other, GraphNode):
            return False
        return self.symbol == other.symbol


@dataclass
class GraphEdge:
    """
    Represents an edge in the symbol graph.
    Connects two symbols with a relationship type and confidence.
    """
    source: GraphNode
    target: GraphNode
    edge_type: EdgeType
    confidence: float = 1.0  # Geometric confidence or MLP probability
    is_geometric: bool = True  # True if from geometric rules, False if from MLP
    
    def __hash__(self):
        return hash((id(self.source), id(self.target), self.edge_type))
    
    def __eq__(self, other):
        if not isinstance(other, GraphEdge):
            return False
        return (self.source == other.source and 
                self.target == other.target and 
                self.edge_type == other.edge_type)


class SymbolGraph:
    """
    Main graph class for managing symbol nodes and their relationships.
    Provides graph construction, query, and traversal APIs.
    """
    
    def __init__(self):
        self.nodes: List[GraphNode] = []
        self.edges: List[GraphEdge] = []
        self.node_map: Dict[Symbol, GraphNode] = {}  # Symbol -> GraphNode mapping
        self.adjacency: Dict[GraphNode, List[GraphEdge]] = {}  # Adjacency list
        
    def add_node(self, symbol: Symbol, node_type: Optional[NodeType] = None) -> GraphNode:
        """
        Adds a symbol as a node in the graph.
        
        Args:
            symbol: The symbol to add
            node_type: Optional node type. If None, inferred from class_name.
            
        Returns:
            The created GraphNode
        """
        if symbol in self.node_map:
            return self.node_map[symbol]
        
        if node_type is None:
            node_type = self._infer_node_type(symbol)
        
        node = GraphNode(
            symbol=symbol,
            node_type=node_type,
            node_id=len(self.nodes)
        )
        
        self.nodes.append(node)
        self.node_map[symbol] = node
        self.adjacency[node] = []
        
        return node
    
    def add_edge(self, source: Symbol, target: Symbol, 
                 edge_type: EdgeType, confidence: float = 1.0,
                 is_geometric: bool = True) -> Optional[GraphEdge]:
        """
        Adds an edge between two symbols.
        
        Args:
            source: Source symbol
            target: Target symbol
            edge_type: Type of relationship
            confidence: Confidence score (0.0-1.0)
            is_geometric: Whether this is a geometric rule-based edge
            
        Returns:
            The created GraphEdge, or None if nodes don't exist
        """
        source_node = self.node_map.get(source)
        target_node = self.node_map.get(target)
        
        if source_node is None or target_node is None:
            return None
        
        # Check if edge already exists
        for edge in self.edges:
            if (edge.source == source_node and edge.target == target_node and 
                edge.edge_type == edge_type):
                # Update confidence if new one is higher
                if confidence > edge.confidence:
                    edge.confidence = confidence
                    edge.is_geometric = is_geometric
                return edge
        
        edge = GraphEdge(
            source=source_node,
            target=target_node,
            edge_type=edge_type,
            confidence=confidence,
            is_geometric=is_geometric
        )
        
        self.edges.append(edge)
        self.adjacency[source_node].append(edge)
        self.adjacency[target_node].append(edge)  # Undirected graph
        
        return edge
    
    def get_neighbors(self, node: GraphNode, edge_type: Optional[EdgeType] = None) -> List[GraphNode]:
        """
        Gets all neighbors of a node, optionally filtered by edge type.
        
        Args:
            node: The node to get neighbors for
            edge_type: Optional filter for edge type
            
        Returns:
            List of neighbor nodes
        """
        neighbors = []
        for edge in self.adjacency.get(node, []):
            if edge_type is None or edge.edge_type == edge_type:
                if edge.source == node:
                    neighbors.append(edge.target)
                else:
                    neighbors.append(edge.source)
        return list(set(neighbors))  # Remove duplicates
    
    def get_edges(self, node: GraphNode, edge_type: Optional[EdgeType] = None) -> List[GraphEdge]:
        """
        Gets all edges connected to a node, optionally filtered by edge type.
        
        Args:
            node: The node to get edges for
            edge_type: Optional filter for edge type
            
        Returns:
            List of edges
        """
        edges = self.adjacency.get(node, [])
        if edge_type is None:
            return edges
        return [e for e in edges if e.edge_type == edge_type]
    
    def find_node_by_symbol(self, symbol: Symbol) -> Optional[GraphNode]:
        """Finds a GraphNode by its Symbol."""
        return self.node_map.get(symbol)
    
    def get_notehead_clusters(self) -> List[List[GraphNode]]:
        """
        Groups noteheads with their connected components (stems, flags, beams, dots).
        Returns clusters of nodes that form complete notes.
        
        Returns:
            List of clusters, each cluster is a list of GraphNodes forming a note
        """
        clusters = []
        visited: Set[GraphNode] = set()
        
        # Find all notehead nodes
        notehead_nodes = [n for n in self.nodes if n.node_type == NodeType.NOTEHEAD]
        
        for notehead in notehead_nodes:
            if notehead in visited:
                continue
            
            # BFS to find all connected components
            cluster = []
            queue = [notehead]
            visited.add(notehead)
            
            while queue:
                current = queue.pop(0)
                cluster.append(current)
                
                # Get all neighbors (stems, flags, beams, dots, accidentals)
                neighbors = self.get_neighbors(current)
                for neighbor in neighbors:
                    if neighbor not in visited:
                        # Only include relevant types
                        if neighbor.node_type in [NodeType.STEM, NodeType.FLAG, 
                                                   NodeType.BEAM, NodeType.DOT, 
                                                   NodeType.ACCIDENTAL]:
                            visited.add(neighbor)
                            queue.append(neighbor)
            
            if cluster:
                clusters.append(cluster)
        
        return clusters
    
    def _infer_node_type(self, symbol: Symbol) -> NodeType:
        """Infers node type from symbol class name."""
        name_lower = symbol.class_name.lower()
        
        if 'notehead' in name_lower or 'note' in name_lower:
            return NodeType.NOTEHEAD
        elif 'stem' in name_lower:
            return NodeType.STEM
        elif 'flag' in name_lower:
            return NodeType.FLAG
        elif 'beam' in name_lower:
            return NodeType.BEAM
        elif 'dot' in name_lower:
            return NodeType.DOT
        elif any(t in name_lower for t in ['sharp', 'flat', 'natural', 'accidental']):
            return NodeType.ACCIDENTAL
        elif 'clef' in name_lower:
            return NodeType.CLEF
        elif 'barline' in name_lower:
            return NodeType.BARLINE
        elif 'rest' in name_lower:
            return NodeType.REST
        else:
            return NodeType.OTHER
    
    def build_from_symbols(self, symbols: List[Symbol]) -> None:
        """
        Builds the graph from a list of symbols.
        Creates nodes for all symbols.
        
        Args:
            symbols: List of symbols to add as nodes
        """
        for symbol in symbols:
            self.add_node(symbol)
    
    def get_statistics(self) -> Dict[str, any]:
        """
        Returns statistics about the graph.
        
        Returns:
            Dictionary with graph statistics
        """
        node_type_counts = {}
        edge_type_counts = {}
        
        for node in self.nodes:
            node_type_counts[node.node_type.value] = node_type_counts.get(node.node_type.value, 0) + 1
        
        for edge in self.edges:
            edge_type_counts[edge.edge_type.value] = edge_type_counts.get(edge.edge_type.value, 0) + 1
        
        return {
            'num_nodes': len(self.nodes),
            'num_edges': len(self.edges),
            'node_types': node_type_counts,
            'edge_types': edge_type_counts,
            'geometric_edges': sum(1 for e in self.edges if e.is_geometric),
            'mlp_edges': sum(1 for e in self.edges if not e.is_geometric)
        }

