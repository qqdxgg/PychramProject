import sys
from typing import List
import random


class Solution:
    def __init__(self, nums: List[int]):
        self.nums = nums;

    def pick(self, target: int) -> int:
        index = 0
        n = 0
        for i in range(0, len(self.nums)):
            if self.nums[i] == target:
                n+=1
                r = random.randint(1,n)
                if r % n == 0: index = i
        return index

if __name__ =="__main__":
    nums = [1,2,3,3,3]
    s = Solution(nums)
    print(s.pick(3))