
import torch
import torch
import torch.fx
from torch.fx.node import Node
from typing import Dict
from networkx.drawing.nx_agraph import to_agraph
import networkx as nx
import gc

active = 25
sample = 5
debug = False

def log(*args, **kwargs):
    if debug:
        print(*args, **kwargs)

def multiplyList(myList):
    # Multiply elements one by one
    result = 1
    for x in myList:
        result = result * x
    return result

def find_accgrad_shapes(node, result, kernel):

    # fill accgrad shapes
    accgrad_shapes = []
    if ("addmm" in node.name) or ("nn.modules.linear.Linear" in str(node.meta)):
        if len(node.input_shapes) == 1:
            op1 = node.input_shapes[0]
        else:
            assert(len(node.input_shapes) == 3)
            bias, op1, op2 = node.input_shapes
        out = result.shape # output shape
        B = 1
        M = multiplyList(op1) / op1[-1]
        N = op1[-1]
        K = out[-1]

        accgrad_shapes.append(torch.Size((K, N))) # weight, transposed
        accgrad_shapes.append(torch.Size((N, K))) # weight
        accgrad_shapes.append(torch.Size((K,))) # bias
        log(f"accgrad_shapes : {accgrad_shapes}")

    elif "embedding" in (str(kernel).lower()):
        output_shape = result.shape # output shape
        H = output_shape[-1]
        if not hasattr(kernel, "num_embeddings"):
            log("no num_embeddings")
        else:
            vocab_size = kernel.num_embeddings # wpe, wte, token_type
            accgrad_shapes.append(torch.Size((vocab_size, H))) # embedding table

    elif "layernorm" in (str(kernel).lower()):
        output_shape = result.shape # output shape
        H = output_shape[-1]
        accgrad_shapes.append(torch.Size((H,))) # mean
        accgrad_shapes.append(torch.Size((H,))) # var

    elif "getattr" in (str(kernel).lower()) and "weight" in node.name:
        accgrad_shapes.append(torch.Size(output_shape))
    
    return accgrad_shapes

def run_kernel(node, kernel, args, kwargs, measure_time, backward):
    gc.collect()
    torch.cuda.empty_cache()
    
    log(f"name : {node.name}")
    log(f"kernel : {kernel}")

    if measure_time:
        device = "cuda"
    else:
        device = "cpu"

    try:
        start = None
        end = None
        if measure_time:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

        # detach for per-operator backward pass measurement, and move to GPU
        # collect AccumulateGrad dims
        new_args = []
        log(f"args len : {len(args)}")
        for a in args:
            if isinstance(a, torch.Tensor):

                na = a.detach().to(device)
                na.requires_grad = a.requires_grad
                # if a.is_leaf:
                    # accum_grad_dims.append(a.shape)

                # clone the arguments for setitem
                if "setitem" in node.name:
                    na = na.clone()

                new_args.append(na)
                log(f"tensor_arg : {na.shape}")
            else:
                log(f"non_tensor_arg : {a}")
                new_args.append(a)

        result = kernel(*new_args, **kwargs)
        if not measure_time:
            return result, 0, 0, 0

        # forward pass
        log("forward pass")
        fw_latency_list = []
        for _ in range(active):
            ()
            result = None
            start.record()
            result = kernel(*new_args, **kwargs)
            end.record()
            torch.cuda.synchronize()
            fw_latency_list.append(start.elapsed_time(end))
        fw_latency_list.sort()
        fw_latency_list = fw_latency_list[:sample]
        fw_latency = sum(fw_latency_list)/len(fw_latency_list)
        if not isinstance(result, torch.Tensor):
            log("non-Tensor result")
            log(result)

        # backward pass
        log("backward pass")
        bw_latency = 0
        acc_latency = 0

        if backward:
            if ("getitem" in node.name) or (node.op == "call_method"):
                log("skip bw")
            elif ("split" in node.name) or (hasattr(result, "grad_fn") and result.grad_fn is not None):
                # traverse grad_fn
                if "split" in node.name:
                    grad_fn = result[0].grad_fn # take first one for grad_fn
                    num_splits = len(result)
                    intermediate = []
                    for _ in range(num_splits):
                        intermediate.append(torch.ones_like(result[0], device="cuda", requires_grad=result[0].requires_grad))
                    intermediate = tuple(intermediate)
                else:
                    grad_fn = result.grad_fn
                    intermediate = torch.ones_like(result, device="cuda", requires_grad=result.requires_grad)

                grad_fns = [grad_fn]
                intermediate_shapes = [intermediate.shape]
                accgrad_shapes = find_accgrad_shapes(node=node, result=result, kernel=kernel)

                while True:
                    if len(grad_fns) == 0:
                        break

                    grad_fn = grad_fns.pop(0)
                    shapes = intermediate_shapes.pop(0)
                
                    if grad_fn is None:
                        break
                    
                    if ("setitem" in  node.name) and ("CloneBackward" in str(grad_fn.name)):
                        # skip CloneBackward for setitem
                        continue
                
                    if isinstance(shapes, torch.Size):
                        intermediate = torch.ones(shapes, device="cuda")
                    else:
                        intermediate = [torch.ones(s, device="cuda") for s in shapes]

                    log("--------------------")
                    log("grad_fn")
                    log(grad_fn)
                    log("intermediate")
                    if (isinstance(intermediate, tuple)):
                        for i in intermediate:
                            log(i.shape)
                    else:
                        log(intermediate.shape)

                    if "AccumulateGrad" in str(grad_fn.name):
                        if not intermediate.shape in accgrad_shapes:
                            # print(f"AccumulateGrad shape not found : {intermediate.shape}")
                            continue
                        
                        try:
                            if isinstance(intermediate, tuple):
                                start.record()
                                _ = grad_fn(*intermediate)
                                end.record()
                            else:
                                start.record()
                                _ = grad_fn(intermediate)
                                end.record()
                        except Exception as e:
                            # match by incident
                            if "doesn't match the broadcast shape" in str(e):
                                continue
                            else:
                                raise e

                    latency_list = []
                    for i in range(active):
                        if isinstance(intermediate, tuple):
                            start.record()
                            _ = grad_fn(*intermediate)
                            end.record()
                        else:
                            start.record()
                            _ = grad_fn(intermediate)
                            end.record()
                        torch.cuda.synchronize()
                        latency_list.append(start.elapsed_time(end))
                    latency_list.sort()
                    latency_list = latency_list[:sample]
                    latency = sum(latency_list)/len(latency_list)

                    if "AccumulateGrad" in str(grad_fn.name):
                        acc_latency += latency
                        # print(acc_latency)
                    else:
                        bw_latency += latency

                    if isinstance(intermediate, tuple):
                        start.record()
                        intermediate = grad_fn(*intermediate)
                        end.record()
                    else:
                        start.record()
                        intermediate = grad_fn(intermediate)
                        end.record()

                    log("grad_fn.next_functions")
                    log(grad_fn.next_functions)

                    next_functions = [t[0] for t in grad_fn.next_functions]
                    grad_fns.extend(next_functions)

                    if (isinstance(intermediate, tuple)):
                        shapes = [i.shape if i is not None else tuple() for i in intermediate]
                        intermediate_shapes.extend(shapes)
                    else:
                        intermediate_shapes.append(intermediate.shape if intermediate is not None else tuple())

                    del intermediate
                    del grad_fn

                grad_fn = None
                intermediate = None
            else:
                log("missing grad_fn")
        
        return result, fw_latency, bw_latency, acc_latency

    except Exception as e:
        log(f"Failed to measure {node.name}")
        raise e
        return result, 0, 0, 0

class NodeProp:
    def __init__(self, mod):
        self.mod = mod
        self.graph = mod.graph
        self.modules = dict(self.mod.named_modules())

    def propagate(self, *args, backward, bench):
        args_iter = iter(args)
        env : Dict[str, Node] = {}
        # kernels  = {}

        def load_arg(a, user_name):
            # print("load arg : ", a)
            def search(n):
                found = env[n.name][0] 
                env[n.name][1][user_name] = True
                if (found is None):
                    print(n.name)
                    print(env[n.name])
                    assert(0)
                return found
            arg = torch.fx.graph.map_arg(a, search)
            return arg

        def fetch_attr(target : str):
            target_atoms = target.split('.')
            attr_itr = self.mod
            for i, atom in enumerate(target_atoms):
                if not hasattr(attr_itr, atom):
                    raise RuntimeError(f"Node referenced nonexistant target {'.'.join(target_atoms[:i])}")
                attr_itr = getattr(attr_itr, atom)
            return attr_itr

        grad_fns = {}

        # annotate latency
        for node in self.graph.nodes:

            # print(node)
            # print("memory allocated : ", torch.cuda.memory_allocated(device=None) / 1024 / 1024 / 1024)

            # Shape
            input_shapes = []
            input_contiguous = []
            for n in node.all_input_nodes:
                if hasattr(n, "output_shape"):
                    input_shapes.append(list(n.output_shape))
                    input_contiguous.append(n.contiguous)
            node.input_shapes = input_shapes
            node.input_contiguous = input_contiguous

            # log("")
            # print("here node")
            # print(node)
            # print("")

            # print("cur node name  : ", node.name)
            # print("cur node arg : ", node.args)

            if node.op == 'placeholder':
                kernel = None
                result = next(args_iter)
                fw_latency = 0
                bw_latency = 0
                acc_latency = 0
            elif node.name == 'output':
                kernel = None
                fw_latency = 0
                bw_latency = 0
                acc_latency = 0
            elif node.op == 'get_attr':
                kernel = None
                result = fetch_attr(node.target)
                fw_latency = 0
                bw_latency = 0
                acc_latency = 0
            elif node.op == 'call_function':
                log("call function")
                kernel = node.target
                result, fw_latency, bw_latency, acc_latency = run_kernel(node, node.target, load_arg(node.args, node.name), load_arg(node.kwargs, node.name), measure_time=bench, backward=backward)
            elif node.op == 'call_method':
                log("call method")
                self_obj, *args = load_arg(node.args, node.name)
                kernel = getattr(self_obj, node.target)
                kwargs = load_arg(node.kwargs, node.name)
                result, fw_latency, bw_latency, acc_latency = run_kernel(node, getattr(self_obj, node.target), args, kwargs, measure_time=bench, backward=backward)
            elif node.op == 'call_module':
                log("call module")
                kernel = self.modules[node.target]
                result, fw_latency, bw_latency, acc_latency = run_kernel(node, self.modules[node.target], load_arg(node.args, node.name), load_arg(node.kwargs, node.name), measure_time=bench, backward=backward)

            # garbage collection on env
            for key in env.keys():
                # print("!!!")
                # for key in env.keys():
                #     print(key)
                #     print(env[key][1])
                # print("!!!")
                if all(env[key][1].values()):
                    # print("collected : ", key)
                    env[key][0] = None

            users = node.users.keys()
            users = list(map(lambda n : n.name, users))
            use_dict = dict(zip(users, [False]*len(users)))
            env[node.name] = [result, use_dict]
            # kernels[str(node.name)] = (node, kernel)
            if hasattr(result, "is_contiguous"):
                node.contiguous = result.is_contiguous()
            else:
                node.contiguous = True

            if bench:
                node.latency = fw_latency + bw_latency + acc_latency
                node.fw_latency = fw_latency
                node.bw_latency = bw_latency
                node.acc_latency = acc_latency

            if isinstance(result, torch.Tensor):
                if result.shape == torch.Size([]):
                    node.output_shape = (1,)
                else:
                    node.output_shape = tuple(result.shape)
                node.dtype = result.dtype
            else:
                node.output_shape = (1,)
            
        return self.mod

def visualize(graph):
    # create graph
    nx_graph = nx.DiGraph()

    nodes_to_hide = [] # for visibility

    for node in graph.nodes:
        idx_node = list(graph.nodes).index(node)
        # add nodes to the graph
        node_tuple = (idx_node, {"node": node, "idx" : idx_node})
        nx_graph.add_nodes_from([node_tuple])
        # add edges to the graph
        for user in node.users:
            nx_graph.add_edge(idx_node, list(
                graph.nodes).index(user),)

    for n in nx_graph.nodes:
        node = nx_graph.nodes[n]["node"]
        nx_graph.nodes[n]["output_shape"] = "box"
        nx_graph.nodes[n]["style"] = "filled"
        nx_graph.nodes[n]["label"] = node.name
        if hasattr(node, "latency"):
            nx_graph.nodes[n]["label"] += f" ({node.latency:.3f}) ms"

    for n in nx_graph.edges:
        edge = nx_graph.edges[n]
        prev_node = nx_graph.nodes[n[0]]["node"]
        if hasattr(prev_node, "output_shape"):
            edge["label"] = str(list(prev_node.output_shape))

    agraph = to_agraph(nx_graph)
    agraph.layout("dot")
    out_filename = "./fx.png"
    agraph.draw(out_filename)
    # display(Image(filename=out_filename))
