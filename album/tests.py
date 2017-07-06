from django.test import TestCase, RequestFactory

from album.models import Album
from album.views import AlbumList
from functional_tests.factory import TalkingAlbumSlideFactory, ImageFactory, AuthorFactory, LocationFactory, \
    AlbumFactory, PhotoAlbumSlideFactory


class TalkingAlbumSlideTest(TestCase):
    def setUp(self):
        author1 = AuthorFactory.create(name='Test1')
        author2 = AuthorFactory.create(name='Test2')
        author3 = AuthorFactory.create(name='Test3')
        location1 = LocationFactory.create(name="Madurai", slug="madurai")
        image1 = ImageFactory.create(photographers=(author1, author2,), locations=(location1,))
        image2 = ImageFactory.create(photographers=(author1, author3,), locations=(location1,))
        self.talking_album = TalkingAlbumSlideFactory(image=image1, page__title="Talking Album Test")
        self.talking_album_1 = TalkingAlbumSlideFactory(page__title="Talking Album Lokesh")
        self.talking_album2 = TalkingAlbumSlideFactory(image= image1, album_title='Talking New')

    def test_title_of_the_talking_album_should_be_equal_to_the_toStr(self):
        assert self.talking_album.page.title == str(self.talking_album.page)


class AlbumListTest(TestCase):
    def setUp(self):
        author1 = AuthorFactory.create(name='Test1')
        author2 = AuthorFactory.create(name='Test2')
        author3 = AuthorFactory.create(name='Test3')
        location1 = LocationFactory.create(name="Madurai", slug="madurai")
        image1 = ImageFactory.create(photographers=(author1, author2,), locations=(location1,))
        self.talking_album1 = TalkingAlbumSlideFactory(image=image1, page__title="Talking Album 1", page__first_published_at='2011-10-24 12:43')
        self.talking_album2 = TalkingAlbumSlideFactory(image=image1, page__title="Talking Album 2", page__first_published_at= '2011-10-25 12:43')
        self.photo_album = PhotoAlbumSlideFactory(image=image1)

    def request_for_albums(self, album_type):
        request = RequestFactory().get('/albums/'+album_type+'/')
        responseSet = AlbumList.as_view()(request, filter=album_type)
        return responseSet

    def test_request_for_talking_url_should_get_talking_album_in_context(self):
        response = self.request_for_albums('talking')
        title = response.context_data['title']
        assert title == 'Talking Albums'

    def test_count_of_talking_albums_created_should_be_two(self):
        responseSet = self.request_for_albums('talking')
        assert len(responseSet.context_data['albums']) == 2

    def test_count_of_photo_albums_created_should_be_two(self):
        response = self.request_for_albums('other')
        assert len(response.context_data['albums']) == 1

    def test_talking_albums_should_come_in_order_of_first_published_date(self):
        response = self.request_for_albums('talking')
        assert response.context_data['albums'][0].title == 'Talking Album 2'

    def test_should_return_count_two_photographer_for_talking_album_2(self):
        response = self.request_for_albums('talking')
        id_of_talking_album1 = self.talking_album1.id
        print id_of_talking_album1
        print response.context_data['photographers']
