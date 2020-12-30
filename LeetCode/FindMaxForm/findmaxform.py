from typing import List


class Solution:

    def findMaxForm(self, strs, m, n):
        def counnum(str):
            res = [0 for i in range(0, 2)]
            for s in str:
                res[ord(s) - ord('0')] += 1
            return res

        dp = [[0 for j in range(0, n + 1)] for i in range(0, m + 1)]
        for str in strs:
            count = counnum(str)
            zeros = m
            while zeros >= count[0]:
                ones = n
                while ones >= count[1]:
                    dp[zeros][ones] = max(dp[zeros][ones], dp[zeros - count[0]][ones - count[1]] + 1)
                    ones -= 1
                zeros -= 1
        return dp[m][n]


if __name__ == '__main__':
    s, m, n = ['10', '0001', '111001', '1', '0'], 5, 3
    # s, m, n = ["10", "0", "1"], 1, 1
    so = Solution()
    print(so.findMaxForm(s, m, n))
