import java.util.PriorityQueue;

import org.junit.Test;

public class NearestK {
	public static Point[] findNearestK(Point[] arr) {
		PriorityQueue<Point> pq = new PriorityQueue<Point>();
		Point p1 = new Point();
		p1.x = 1;
		p1.y = 1;
		Point p2 = new Point();
		p2.x = -1;
		p2.y = -2;
		Point p3 = new Point();
		p3.x = 0.5;
		p3.y = 0.5;
		pq.add(p1);
		pq.add(p2);
		pq.add(p3);
		System.out.println(pq.poll());
		System.out.println(pq.poll());
		System.out.println(pq.poll());
		return null;
	}
	
	@Test
	public void testFind() {
		findNearestK(null);
	}
}

class Point implements Comparable<Point> {
	public double x;
	public double y;

	@Override
	public int compareTo(Point o) {
		double d1 = Math.sqrt(x * x + y * y);
		double d2 = Math.sqrt(o.x * o.x + o.y * o.y);
		
		if (d1 > d2) {
			return 1;
		} else if (d1 == d2) {
			return 0;
		} else {
			return -1;
		}
	}
	
	@Override
	public String toString() {
		return x + " " + y;
	}
}
