
public class test {
	public static void main(String[] args) {
		int i = Integer.MIN_VALUE;
		System.out.println(Integer.toBinaryString(i));
		int j = ~i + 1;
		System.out.println(j);
	}

}
