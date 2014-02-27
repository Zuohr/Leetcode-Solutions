import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.junit.Test;

public class Q3 {
	Map<Integer, Double> calculateFinalScores(List<TestResult> results) {
		Map<Integer, Double> result = new HashMap<Integer, Double>();
		if (results == null) {
			return result;
		}

		Map<Integer, ArrayList<TestResult>> scoreMap = new HashMap<Integer, ArrayList<TestResult>>();
		for (TestResult testResult : results) {
			ArrayList<TestResult> scores = scoreMap.get(testResult.studentId);
			if (scores == null) {
				scores = new ArrayList<TestResult>();
				scoreMap.put(testResult.studentId, scores);
			}
			insertScore(scores, testResult);
		}

		for (Integer id : scoreMap.keySet()) {
			ArrayList<TestResult> scores = scoreMap.get(id);
			double sum = 0;
			int count = 0;
			for (TestResult testResult : scores) {
				sum += testResult.testScore;
				count++;
			}
			result.put(id, sum / count);
		}

		return result;
	}

	void insertScore(List<TestResult> scoreList, TestResult testResult) {
		if (scoreList.size() == 5) {
			TestResultComparator cmp = new TestResultComparator();
			for (int i = 0; i < scoreList.size(); i++) {
				if (cmp.compare(scoreList.get(i), testResult) > 0) {
					for (int j = 4; j > i; j--) {
						scoreList.set(j, scoreList.get(j - 1));
					}
					scoreList.set(i, testResult);
					break;
				}
			}
		} else {
			scoreList.add(testResult);
			if (scoreList.size() == 5) {
				Collections.sort(scoreList, new TestResultComparator());
			}
		}
	}
	
	@Test
	public void test() {
		ArrayList<TestResult> score = new ArrayList<TestResult>();
		for (int i = 0; i < 10; i++) {
			for (int j = 0; j < 10; j++) {
				TestResult r = new TestResult();
				r.studentId = i;
				r.testScore = j;
				score.add(r);
			}
		}
		Map<Integer, Double> result = calculateFinalScores(score);
		for (Integer key : result.keySet()) {
			double avg = result.get(key);
			System.out.println(key + " " + avg);
		}
	}
}

class TestResult {
	int studentId;
	String testDate;
	int testScore;
}

class TestResultComparator implements Comparator<TestResult> {
	// sort descending
	@Override
	public int compare(TestResult o1, TestResult o2) {
		return o2.testScore - o1.testScore;
	}

}
