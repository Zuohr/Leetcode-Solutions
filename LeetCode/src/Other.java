import java.util.ArrayList;
import java.util.HashSet;

import org.junit.Test;

public class Other {
	public int[] markExistence(int[] arr) {
		if (arr == null) {
			return arr;
		}

		int len = arr.length;
		for (int i = 0; i < len; i++) {
			while (i != arr[i] && arr[i] != arr[arr[i]]) {
				int j = arr[i];
				arr[i] ^= arr[j];
				arr[j] ^= arr[i];
				arr[i] ^= arr[j];
			}
		}
		
		for (int i = 0; i < len; i++) {
			if (i != arr[i]) {
				arr[i] = -1;
			}
		}

		return arr;
	}
	
	public void findDuplicates(int[] arr) {
		if (arr != null) {
			int len = arr.length;
			for (int i = 0; i < len; i++) {
				if (arr[Math.abs(arr[i])] > 0) {
					arr[Math.abs(arr[i])] *= -1;
				} else {
					System.out.println(Math.abs(arr[i]));
				}
			}
		}
	}
	
	/*
	 * Find duplicated elements in two arrays
	 */

	// case 1: sorted, O(1) space
	public ArrayList<Integer> findDupInTwoArrays1(int[] arr1, int[] arr2) {
		ArrayList<Integer> result = new ArrayList<Integer>();
		if (arr1 == null || arr1.length == 0 || arr2 == null || arr2.length == 0) {
			return result;
		}

		int len1 = arr1.length, len2 = arr2.length;
		int p1 = 0, p2 = 0;
		while (p1 < len1 && p2 < len2) {
			if (arr1[p1] == arr2[p2]) {
				result.add(arr1[p1]);
			} else if (arr1[p1] > arr2[p2]) {
				p2++;
			} else {
				p1++;
			}
		}

		return result;
	}

	// case 2: O(n) space, O(n) time
	public HashSet<Integer> findDupInTwoArrays2(int[] arr1, int[] arr2) {
		HashSet<Integer> result = new HashSet<Integer>();
		if (arr1 == null || arr1.length == 0 || arr2 == null || arr2.length == 0) {
			return result;
		}

		HashSet<Integer> set = new HashSet<Integer>();
		for (int elm : arr1) {
			set.add(elm);
		}
		for (int elm : arr2) {
			if (set.contains(elm)) {
				result.add(elm);
			}
		}

		return result;
	}
	
	@Test
	public void testFindDuplicates() {
		int[] arr = { 2, 2, 1, 2, 3, 1, 3, 6, 6, 0, 6, 6 };
		findDuplicates(arr);
	}
}
