import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

public class LineCounter {

	public static void main(String[] args) {
		if (args.length > 0) {
			File file = new File(args[0]);
			if (file.exists() && file.isDirectory()) {
				int num = countLine(file);
				System.out.println(num);
			}
		}
	}

	public static int countLine(File file) {
		int num = 0;
		if (file != null) {
			File[] files = file.listFiles();
			for (File f : files) {
				if (f.isDirectory()) {
					num += countLine(f);
				} else if (f.isFile() && f.getAbsolutePath().endsWith(".java")
						|| f.getAbsolutePath().endsWith(".jsp")
						|| f.getAbsolutePath().endsWith(".jspf")) {
					num += countFileLine(f);
				}
			}
		}
		return num;
	}

	private static int countFileLine(File f) {
		int num = 0;
		System.out.println("reading " + f.getAbsolutePath());
		try {
			BufferedReader r = new BufferedReader(new FileReader(f));
			while(r.readLine() != null) {
				num++;
			}
			r.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		return num;
	}

}
