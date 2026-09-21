import pandas as pd

class Node:
    def __init__(self, row) -> None:
        self.Name = row["Name"]
        self.OpName = row["OpName"]
        self.FwOps = row["FwOps"]
        self.BwOps = row["BwOps"]
        self.AccOps = row["AccOps"]
        self.PrevName = row["Prev"]
        self.Prev = None
        self.NextName = row["Next"]
        self.Next = None
        self.InputShape = None
        self.OutputShape = row["OutputShape"]
    
    def dump(self):
        return {
            "Name": self.Name,
            "OpName": self.OpName,
            "FwOps": self.FwOps,
            "BwOps": self.BwOps,
            "AccOps": self.AccOps,
            "Prev": {n.Name for n in self.Prev},
            "Next": {n.Name for n in self.Next},
            "InputShapes": self.InputShape,
            "OutputShape": self.OutputShape,
        }
    
    def fuse(self, other):
        old_name = self.Name
        self.Name = self.Name + "~" + other.Name

        # fuse only fw for now
        self.FwOps = self.FwOps + other.FwOps
        
        # fuse only fw for now
        self.BwOps += other.BwOps
        self.AccOps += other.AccOps
        self.OutputShape = other.OutputShape

        self.OpName = "fused"

        self.Next = other.Next
        for n in other.Next:
            n.Prev = {self if x == other else x for x in n.Prev}
        for n in other.Prev:
            if n == self:
                continue
            n.Next = {self if x == other else x for x in n.Next}
            self.Prev.add(n)

        # print(old_name)
        return old_name

class OpGraph:
    def __init__(self, df) -> None:
        self.df = df
        self.nodes = [] # topologically sorted
        self.nodes_dict = {}
        self.build_graph()
    
    def build_graph(self):
        for idx, row in self.df.iterrows():
            node = Node(row)
            self.nodes.append(node)
            self.nodes_dict[node.Name] = node
        
        for node in self.nodes:
            node.Prev = {self.nodes_dict[name] for name in node.PrevName}
            delattr(node, "PrevName")
            node.Next = {self.nodes_dict[name] for name in node.NextName}
            delattr(node, "NextName")
    
    def dump_df(self):
        entries = []
        for node in self.nodes:
            entries.append(node.dump())
        df = pd.DataFrame(entries)
        return df
    
    def trace_input_shapes(self):
        for node in self.nodes:
            node.InputShape = [n.OutputShape for n in node.Prev]
    
    def fuse_nodes(self, node1 : Node, node2 : Node):
        old_name = node1.fuse(node2)

        # remove node2
        self.nodes.remove(node2)
        del self.nodes_dict[node2.Name]

        # update node1 with new name
        del self.nodes_dict[old_name]
        self.nodes_dict[node1.Name] = node1

    def fuse(self):
        def is_fusable(node):
            is_vec_op = lambda node : (node.OpName.startswith("VEC") or (node.OpName in ["fused", "MEM", "EMBEDDING", "dropout", "misc"]))
            return len(node.Next) == 1 and is_vec_op(node) and is_vec_op(list(node.Next)[0])

        while True:
            fused = False
            idx = 0
            while idx < len(self.nodes):
                node = self.nodes[idx]
                if is_fusable(node):
                    self.fuse_nodes(node, list(node.Next)[0])
                    fused = True
                else:
                    idx += 1
            if not fused:
                break

def fuse_parse(input_csv):
    opgraph = OpGraph(input_csv)
    opgraph.fuse()
    opgraph.trace_input_shapes()
    df = opgraph.dump_df()
    return df