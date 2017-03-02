import itertools

from django.conf import settings
from django.core.urlresolvers import reverse
from django.utils.text import slugify
from django.views.generic.detail import DetailView
from django.views.generic.list import ListView
from django.db.models import Q
from wagtail.wagtailadmin.modal_workflow import render_modal_workflow
from itertools import chain

from album.models import Album
from album.models import AlbumSlide
from article.models import Article
from core.models import AffixImage
from face.models import Face
from .forms import LocationAdminForm
from .models import Location


class LocationList(ListView):
    context_object_name = "locations"
    model = Location
    template_name = "map/map_list.html"

    def get_queryset(self):
        qs = super(LocationList, self).get_queryset()
        location_filter = self.request.GET.getlist("filter", [])
        if location_filter:
            location_ids = []
            if "articles" in location_filter:
                articles = Article.objects.live()
                location_ids.extend(articles.values_list('locations', flat=True).distinct())
            if "albums" in location_filter:
                slides = AlbumSlide.objects.all().select_related('image').filter(page__in=Album.objects.live(), image__isnull=False)
                locations_of_images_of_live_albums = map((lambda slide: slide.image.locations.values_list('id', flat=True)), slides)
                location_ids.extend(chain(*locations_of_images_of_live_albums))
            if "faces" in location_filter:
                faces = Face.objects.live()
                location_ids.extend(faces.values_list('location', flat=True).distinct())
                locations_of_image_of_live_faces = map((lambda face: face.image.locations.values_list('id', flat=True)), faces)
                location_ids.extend(chain(*locations_of_image_of_live_faces))
            qs = qs.filter(id__in=location_ids)
        return qs

    def get_context_data(self, **kwargs):
        context = super(LocationList, self).get_context_data(**kwargs)
        req = self.request
        filters = req.GET.getlist("filter")
        if filters:
            context["articles_checked"] = "articles" in filters
            context["albums_checked"] = "albums" in filters
            context["faces_checked"] = "faces" in filters
        else:
            context["articles_checked"], context["albums_checked"], \
                context["faces_checked"] = True, True, True
        context['MAP_KEY'] = settings.GOOGLE_MAP_KEY
        return context


class LocationDetail(DetailView):
    context_object_name = "location"
    model = Location

    def get_context_data(self, **kwargs):
        context = super(LocationDetail, self).get_context_data(**kwargs)
        location = context['location']
        articles_qs = Article.objects.live().filter(locations=location)
        images_qs = AffixImage.objects.filter(locations=location)
        albums_slides = AlbumSlide.objects.filter(image__in=images_qs)
        albums_qs = Album.objects.live().filter(slides=albums_slides)
        live_faces = Face.objects.live()
        faces_qs = live_faces.filter(Q(location=location) | Q(image__in=images_qs))

        if self.request.GET.get("lang"):
            lang = self.request.GET["lang"]
            articles_qs = articles_qs.filter(language=lang)
            albums_qs = albums_qs.filter(language=lang)
            # faces_qs = faces_qs.filter(language=lang)
        context['articles'] = itertools.chain(
            articles_qs,
            albums_qs,
            faces_qs
        )
        context['LANGUAGES'] = settings.LANGUAGES
        return context


def add_location(request):
    instance = None
    if request.method == "POST":
        form = LocationAdminForm(request.POST)
        if form.is_valid():
            instance = form.save(commit=False)
            instance.slug = slugify(instance.name)[:50]
            instance.save()
    else:
        form = LocationAdminForm()
    return render_modal_workflow(
        request, "location/add_location.html", "location/add_location.js", {
            "add_object_url": reverse("locations_add"),
            "name": "Location",
            "form": form,
            "instance": instance,
            "default_lat": settings.MAP_CENTER[0],
            "default_lon": settings.MAP_CENTER[1],
            "default_zoom": 6
        })
