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

	@Test
	public void testFindDuplicates() {
		int[] arr = { 2, 2, 1, 2, 3, 1, 3, 6, 6, 0, 6, 6 };
		findDuplicates(arr);
	}
}
