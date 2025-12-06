"""
Simple HeteroData implementation for compatibility
"""

class SimpleEdgeData:
    def __init__(self):
        self.edge_index = None
        self.edge_attr = None

class SimpleHeteroData:
    def __init__(self):
        self.x_dict = {}
        self.edge_index_dict = {}
        self.edge_attr_dict = {}
        self.user_id_mapping = {}
        self.food_id_mapping = {}
    
    def __getitem__(self, key):
        """딕셔너리 스타일 접근"""
        if isinstance(key, tuple):
            # Edge 접근
            edge_data = SimpleEdgeData()
            edge_data.edge_index = self.edge_index_dict.get(key)
            edge_data.edge_attr = self.edge_attr_dict.get(key)
            return edge_data
        return getattr(self, key, None)
    
    def to(self, device):
        """GPU 이동"""
        for key in self.x_dict:
            self.x_dict[key] = self.x_dict[key].to(device)
        for key in self.edge_index_dict:
            self.edge_index_dict[key] = self.edge_index_dict[key].to(device)
        for key in self.edge_attr_dict:
            self.edge_attr_dict[key] = self.edge_attr_dict[key].to(device)
        return self
