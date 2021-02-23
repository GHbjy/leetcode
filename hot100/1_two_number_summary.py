class Solution:
    def twoSum(self, nums: [int], target: int) -> [int]:
        i = 0
        hash_map = {}
        result = []

        while i < len(nums):
            if nums[i] not in hash_map.keys():
                hash_map[nums[i]] = i
            i += 1

            if target - nums[i] in hash_map.keys():
                result.append(hash_map[target - nums[i]])
                result.append(i)
                break

        return result


if __name__ == '__main__':
    # num_list = [2, 7, 11, 15]
    # _target = 9

    # num_list = [3, 2, 4]
    # _target = 6

    num_list = [3, 3]
    _target = 6

    solution = Solution()
    result = solution.twoSum(num_list, _target)
    print('result:', result)
