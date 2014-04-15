import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.PriorityQueue;

import org.junit.Test;

public class MergeKFiles {
	public static void main(String[] args) throws IOException {
		
		int fileNum = 29, cnt = 0;

		BufferedReader[] readers = new BufferedReader[fileNum];
		for (int i = 0; i < fileNum; i++) {
			readers[i] = new BufferedReader(new FileReader(
					String.format("part-000%02d", i)));
		}

		PriorityQueue<Line> pq = new PriorityQueue<Line>();
		for (int i = 0; i < fileNum; i++) {
			String currLine = readers[i].readLine();
			if (currLine != null) {
				Line newLine = new Line(currLine, i);
				pq.add(newLine);
			}
		}

		PrintWriter writer = new PrintWriter(new FileWriter(new File(
				"sorted.txt")));
		while (!pq.isEmpty()) {
			Line curr = pq.poll();
			writer.println(curr.line);
			cnt++;
			if (cnt % 10000 == 0) {
				System.out.println(cnt);
			}
			String nextLine = readers[curr.reader].readLine();
			if (nextLine != null) {
				pq.add(new Line(nextLine, curr.reader));
			}
		}

		// close I/O
		writer.close();
		for (BufferedReader reader : readers) {
			reader.close();
		}
		
		System.out.println(cnt);
	}

	static class Line implements Comparable<Line> {
		String line;
		int reader;

		public Line(String line, int reader) {
			this.line = line;
			this.reader = reader;
		}

		@Override
		public int compareTo(Line l2) {
			return this.line.compareTo(l2.line);
		}
	}

	@Test
	public void test() {
		Line[] arr = { new Line("00000000000560670324\t12", 0),
				new Line("00000000000560670314\t4323", 1),
				new Line("00000000010560670324\t1", 2),
				new Line("00000000000060670980\t12", 3),
				new Line("00000000000561670324\t423", 4) };
		PriorityQueue<Line> pq = new PriorityQueue<MergeKFiles.Line>();
		for (Line l : arr) {
			pq.add(l);
		}
		while (!pq.isEmpty()) {
			System.out.println(pq.poll().line);
		}
	}
}