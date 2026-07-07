"""
本文件借鉴自 hiddenlayer 项目
Licensed under the MIT License
"""

import re

class GEParser():
    """
    图表达式（Graph Expression）解析器。
    支持用类似 "conv > relu | bn" 这样的字符串描述节点模式，并解析为可用于匹配计算图结构的对象。
    """
    def __init__(self, text):
        self.index = 0      # 当前解析位置
        self.text = text    # 输入的表达式字符串

    def parse(self):
        """
        入口方法，尝试按串行、并行、单节点表达式顺序解析。
        """
        return self.serial() or self.parallel() or self.expression()

    def parallel(self):
        """
        解析并行结构（如 "a | b"），返回 ParallelPattern。
        """
        index = self.index
        expressions = []
        while len(expressions) == 0 or self.token("|"):
            e = self.expression()
            if not e:
                break
            expressions.append(e)
        if len(expressions) >= 2:
            return ParallelPattern(expressions)
        # 没有匹配，回退
        self.index = index
    
    def serial(self):
        """
        解析串行结构（如 "a > b > c"），返回 SerialPattern。
        """
        index = self.index
        expressions = []
        while len(expressions) == 0 or self.token(">"):
            e = self.expression()
            if not e:
                break
            expressions.append(e)

        if len(expressions) >= 2:
            return SerialPattern(expressions)
        self.index = index

    def expression(self):
        """
        解析一个表达式，可以是括号包裹的子表达式，也可以是单个操作。
        """
        index = self.index
        
        if self.token("("):
            # 括号内可以是串行、并行或单操作
            e = self.serial() or self.parallel() or self.op()
            if e and self.token(")"):
                return e
        self.index = index
        e = self.op()
        return e

    def op(self):
        """
        解析单个操作（如 conv、relu），可带条件（暂未实现）。
        """
        t = self.re(r"\w+")
        if t:
            c = self.condition()
            return NodePattern(t, c)
    
    def condition(self):
        """
        解析条件（如 [1x1]），目前仅占位，未实现。
        """
        index = self.index
        if self.token("["):
            c = self.token("1x1") or self.token("3x3")
            if c:
                if self.token("]"):
                    return c
            self.index = index
    
    def token(self, s):
        """
        匹配并消费一个特定的符号（如 ">", "|", "(", ")"）。
        """
        return self.re(r"\s*(" + re.escape(s) + r")\s*", 1)

    def string(self, s):
        """
        精确匹配字符串 s。
        """
        if s == self.text[self.index:self.index+len(s)]:
            self.index += len(s)
            return s

    def re(self, regex, group=0):
        """
        用正则表达式从当前位置匹配，成功则推进index。
        """
        m = re.match(regex, self.text[self.index:])
        if m:
            self.index += len(m.group(0))
            return m.group(group)
            

class NodePattern():
    """
    单节点模式，用于匹配某种操作类型的节点。
    """
    def __init__(self, op, condition=None):
        self.op = op
        self.condition = condition  # 目前未用到
    
    def match(self, graph, node):
        """
        判断当前节点是否匹配该模式。
        """
        # 如果是列表，说明有多条边，不允许
        if isinstance(node, list):
            return [], None
        # op类型相同且未被标记跳过
        if self.op == node.op and not node._skip_pattern_search:
            # 只允许单一后继节点
            following = graph.outgoing(node)
            if len(following) == 1:
                following = following[0]
            return [node], following
        else:
            return [], None


class SerialPattern():
    """
    串行模式，匹配一串连续节点（如 conv > relu > bn）。
    """
    def __init__(self, patterns):
        self.patterns = patterns

    def match(self, graph, node):
        """
        顺序匹配每个子模式，要求节点顺序一致。
        """
        all_matches = []
        for i, p in enumerate(self.patterns):
            matches, following = p.match(graph, node)
            if not matches:
                return [], None
            all_matches.extend(matches)
            if i < len(self.patterns) - 1:
                node = following  # 继续下一个节点
        return all_matches, following


class ParallelPattern():
    """
    并行模式，匹配多个兄弟节点（如 conv | bn）。
    """
    def __init__(self, patterns):
        self.patterns = patterns

    def match(self, graph, nodes):
        """
        匹配所有兄弟节点，要求每个pattern都能匹配一个节点，且所有分支最终汇合到同一个节点。
        """
        if not nodes:
            return [], None
        nodes = nodes if isinstance(nodes, list) else [nodes]
        # 如果只有一个节点，获取其所有兄弟节点
        if len(nodes) == 1:
            nodes = graph.siblings(nodes[0])
        else:
            # 检查所有节点的父节点是否一致
            parents = [graph.incoming(n) for n in nodes]
            matches = [set(p) == set(parents[0]) for p in parents[1:]]
            if not all(matches):
                return [], None

        # 节点数和模式数必须一致
        if len(self.patterns) != len(nodes):
            return [], None
        
        patterns = self.patterns.copy()
        nodes = nodes.copy()
        all_matches = []
        end_node = None
        for p in patterns:
            found = False
            for n in nodes:
                matches, following = p.match(graph, n)
                if matches:
                    found = True
                    nodes.remove(n)
                    all_matches.extend(matches)
                    # 检查所有分支最终是否汇合到同一个节点
                    if end_node:
                        if end_node != following:
                            return [], None
                    else:
                        end_node = following
                    break
            if not found:
                return [], None
        return all_matches, end_node


