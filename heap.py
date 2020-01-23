from heapq import heappush, heappushpop, nlargest


class MaxHeap(object):
    def __init__(self, top_n):
        self.h = []
        self.length = top_n

    def push(self, element):
        if len(self.h) < self.length:
            heappush(self.h, element)
        else:
            heappushpop(self.h, element)

    def top(self):
        return nlargest(self.length, self.h)
