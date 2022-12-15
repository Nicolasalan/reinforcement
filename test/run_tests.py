import rosunit
import integration_test

# rosunit
rosunit.unitrun('motion', 'integration_test',
                'integration_test.MyTestSuite')