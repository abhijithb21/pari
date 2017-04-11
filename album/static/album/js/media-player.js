function MediaPlayerControl(mediaElement) {
  this.mediaElement = $(mediaElement);
  this.volume = 0;
  this.time = 0;
  this.isPaused = false;
  this.isMouseDown = false;
  this.startX = 0;

  this.hideAllControls();
  this.createEvents();
}

MediaPlayerControl.prototype.getState = function() {
  return {
    volume: this.volume,
    isPaused: this.isPaused,
    seekPercent: this.seekPercent
  }
}

MediaPlayerControl.prototype.hideAllControls = function() {
  $(this.mediaElement).find(".toggle-tooltip").css("visibility", "hidden");
  // document.body.removeEventListener('click', this.hideAllControls.bind(this));
}

MediaPlayerControl.prototype.createEvents = function() {
  var self = this;
  $(this.mediaElement).find(".play-pause").click(this.handlePlayPauseClick.bind(this));
  $(this.mediaElement).find(".volume-symbol").click(this.handleVolumeClick.bind(this));

  $(this.mediaElement).find(".volume-control .seekbar-max-length").click(this.handleSeekbarClick.bind(this, function(value) {
    this.volume = value;
    $(this.mediaElement).trigger("volume-seekbar-click", this.getState());
  }));

  $(this.mediaElement).find(".seek-bar-control .seekbar-max-length").click(this.handleSeekbarClick.bind(this, function(seekPercent) {
    this.seekPercent = seekPercent;
    $(this.mediaElement).trigger("seeker-seekbar-click", this.getState());
  }));

  $(this.mediaElement).find(".volume-control .seeker").on("mousedown", this.mouseDown.bind(this));
  $(this.mediaElement).find(".volume-control .seeker").on("mousemove", function(event) {
    self.drag(event, function(seekPercent) {
      this.volume = seekPercent;
      $(this.mediaElement).trigger("volume-seekbar-drag", this.getState());
    });
  });
  $(this.mediaElement).find(".volume-control .seeker").on("mouseup", this.mouseUp.bind(this));


  $(this.mediaElement).find(".seek-bar-control .seeker").on("mousedown", this.mouseDown.bind(this));
  $(this.mediaElement).find(".seek-bar-control .seeker").on("mousemove", function(event) {
    self.drag(event, function(seekPercent) {
      this.seekPercent = seekPercent;
      $(this.mediaElement).trigger("seeker-seekbar-drag", this.getState());
    });
  });
  $(this.mediaElement).find(".seek-bar-control .seeker").on("mouseup", this.mouseUp.bind(this));
}

MediaPlayerControl.prototype.handlePlayPauseClick = function(event) {
  event.stopPropagation();
  var element = $(this.mediaElement).find(".play-pause");
  element.toggleClass("selected")
  .removeClass("fa-play")
  .removeClass("fa-pause")
  .addClass(element.hasClass("selected") ? "fa-pause" : "fa-play");
  this.isPaused = element.hasClass("selected");
  $(this.mediaElement).trigger("play-pause-click", this.getState());
}

MediaPlayerControl.prototype.handleVolumeClick = function(event) {
  event.stopPropagation();
  var element = $(this.mediaElement).find(".volume-symbol");
  element.toggleClass("selected");
  if(element.hasClass("selected")) {
    $(this.mediaElement).find(".volume-control .seek-bar").css("visibility", "visible");
    // document.body.addEventListener('click', this.hideAllControls.bind(this));
  } else {
      $(this.mediaElement).find(".volume-control .seek-bar").css("visibility", "hidden");
  }

}

MediaPlayerControl.prototype.handleSeekbarClick = function(callback, event) {
  if($(event.target).hasClass("seeker")) {
    return;
  }
  var rootElement = $(event.target).parents(".control");
  var element = rootElement.find(".seekbar-filled");
  element.css("width", event.offsetX + "px");
  element.find(".seeker").css("left", event.offsetX + "px");
  callback.call(this, parseInt((event.offsetX / rootElement.find(".seekbar-max-length").width()) * 100));
}

MediaPlayerControl.prototype.mouseDown = function(event) {
  event.stopPropagation();
  this.callback = null;
  this.isMouseDown = true;
  this.startX = event.pageX;
  this.draggedElement = event.target;
  document.body.addEventListener('mousemove', this.drag.bind(this));
  document.body.addEventListener('mouseup', this.mouseUp.bind(this));
}

MediaPlayerControl.prototype.drag = function(event, callback) {

  event.stopPropagation();
  if(!this.isMouseDown) {
    return;
  }

  if(callback && !this.callback) {
    this.callback = callback;
  }

  var rootElement = $(this.draggedElement).parents(".control");
  var seekbarParent = rootElement.find(".seekbar-max-length");
  var seekbar = seekbarParent.find(".seeker");
  var left = event.pageX - seekbarParent.offset().left;
  var oldLeft = parseFloat($(seekbar).css("left"));
  var seekbarWidth = seekbarParent.width();

  if(left > seekbarWidth) {
    left = seekbarWidth;
  }

  if(left < 0) {
    left = 0;
  }

  $(seekbar).css("left", left + "px");
  rootElement.find(".seekbar-filled").css("width", left + "px");
  this.callback.call(this, parseInt((left / seekbarWidth) * 100));
  this.startX = event.pageX;
}

MediaPlayerControl.prototype.mouseUp = function(event) {
  document.body.removeEventListener('mousemove', this.drag.bind(this));
  document.body.removeEventListener('mouseup', this.mouseUp.bind(this));
  this.isMouseDown = false;
  this.draggedElement = null;
  this.callback = null;
}

MediaPlayerControl.prototype.setSeek = function(value) {
  var seekbarFilled = $(this.mediaElement).find(".seek-bar-control .seekbar-filled");

  var rootElement = $(event.target).parents(".control");
  var element = rootElement.find(".seekbar-filled");
  element.css("width", event.offsetX + "px");
  element.find(".seeker").css("left", event.offsetX + "px");
  callback.call(this, parseInt((event.offsetX / rootElement.find(".seekbar-max-length").width()) * 100));
}

MediaPlayerControl.prototype.updateOnSeek = function(currentTime, totalTime) {
  var currentTimeString = this.getTime(currentTime);
  var totalTimeString = this.getTime(totalTime);
  var seekPercent = (currentTime / totalTime) * 100;
  var maxBar = $(this.mediaElement).find(".seek-bar-control .seekbar-max-length");
  var seekbarFilled = $(this.mediaElement).find(".seek-bar-control .seekbar-filled");
  var seekInPixels = maxBar.width() * seekPercent / 100;

  seekbarFilled.css("width", seekInPixels + "px");
  seekbarFilled.find(".seeker").css("left", seekInPixels + "px");
  $(".current-length").html(currentTimeString);
  $(".total-length").html(totalTimeString);

}

MediaPlayerControl.prototype.getTime = function(currentTime) {
  var minutes = Math.floor(currentTime / 60000);
  var seconds = ((currentTime % 60000) / 1000).toFixed(0);
  return minutes + ":" + (seconds < 10 ? '0' : '') + seconds;
}

MediaPlayerControl.prototype.setVolume = function(volume) {
  var maxBar = $(this.mediaElement).find(".volume-control .seekbar-max-length");
  var seekbarFilled = $(this.mediaElement).find(".volume-control .seekbar-filled");
  var seekInPixels = maxBar.width() * volume;
  seekbarFilled.css("width", seekInPixels + "px");
  seekbarFilled.find(".seeker").css("left", seekInPixels + "px");
}