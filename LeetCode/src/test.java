
public class test {
	public static void main(String[] args) {
		Test1 t = new Test1();
		System.out.println(t.getNum());
	}

}

class Test1 {
	public int getNum() {
		return num;
	}

	public void setNum(int num) {
		this.num = num;
	}

	private int num;

}
