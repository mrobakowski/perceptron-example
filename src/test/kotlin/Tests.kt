import com.github.mrobakowski.perceptron.eq
import com.github.mrobakowski.perceptron.extendWithColumns
import golem.end
import golem.mapMatIndexed
import golem.mat
import org.junit.Test

class Tests {
    @Test fun testExtend() {
        val m = mat[1, 2, 3 end
                4, 5, 6 end
                7, 8, 9]
        val c = mat[11, 12 end
                13, 14 end
                15, 16]

        val res = mat[1, 2, 3, 11, 12 end
                4, 5, 6, 13, 14 end
                7, 8, 9, 15, 16]
        assert(m.extendWithColumns(c) eq res)
    }
}