import java.util.Arrays;
import java.util.Date;
import java.util.Random;

import org.junit.Test;


public class AverageScore {
	public static Student[] getAverage(Student[] list) {
		
		
		return null;
	}
	
	public static void insert(int[] arr, int n) {
		for (int i = 0; i < arr.length; i++) {
			if (n >= arr[i]) {
				for (int j = arr.length - 1; j > i; j--) {
					arr[j] = arr[j - 1];
				}
				arr[i] = n;
				break;
			}
		}
	}
	
	@Test
	public void testInsert() {
		int[] arr = { 9, 5, 3, 2, 0 };
		Random r = new Random();
		for (int i = 0; i < 20; i++) {
			System.out.print(Arrays.toString(arr));
			int n = r.nextInt(20);
			System.out.printf(" %d ", n);
			insert(arr, n);
			System.out.printf("%s\n", Arrays.toString(arr));
		}
	}
}

class Student {
	public String id;
	public Date date;
	public int score;
}
