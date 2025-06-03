import unittest
from app import app

class BasicTests(unittest.TestCase):

    # executed prior to each test
    def setUp(self):
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        app.config['DEBUG'] = False
        self.app = app.test_client()
        self.assertEqual(app.debug, False)

    # executed after each test
    def tearDown(self):
        pass

    def test_home_route(self):
        response = self.app.get('/', follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Welcome', response.data) # Or some expected text

if __name__ == "__main__":
    unittest.main()