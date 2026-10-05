(function () {
  'use strict';

  document.getElementById('year').textContent = new Date().getFullYear();

  /* ---------------- mobile nav ---------------- */
  var navToggle = document.getElementById('navToggle');
  var mainNav = document.getElementById('mainNav');
  navToggle.addEventListener('click', function () {
    var open = mainNav.classList.toggle('open');
    navToggle.setAttribute('aria-expanded', String(open));
  });
  mainNav.querySelectorAll('a').forEach(function (a) {
    a.addEventListener('click', function () {
      mainNav.classList.remove('open');
      navToggle.setAttribute('aria-expanded', 'false');
    });
  });

  /* ---------------- publication filters ---------------- */
  var pills = Array.prototype.slice.call(document.querySelectorAll('.filter-pill'));
  var items = Array.prototype.slice.call(document.querySelectorAll('.pub-item'));
  var yearGroups = Array.prototype.slice.call(document.querySelectorAll('.pub-year-group'));
  var emptyState = document.getElementById('emptyState');

  function applyFilter(type) {
    var anyVisible = false;
    items.forEach(function (item) {
      var match = type === 'all' || item.getAttribute('data-type') === type;
      item.hidden = !match;
      if (match) anyVisible = true;
    });
    yearGroups.forEach(function (group) {
      var visibleInGroup = group.querySelectorAll('.pub-item:not([hidden])').length > 0;
      group.hidden = !visibleInGroup;
    });
    emptyState.hidden = anyVisible;
  }

  pills.forEach(function (pill) {
    pill.addEventListener('click', function () {
      pills.forEach(function (p) { p.setAttribute('aria-pressed', 'false'); });
      pill.setAttribute('aria-pressed', 'true');
      applyFilter(pill.getAttribute('data-filter'));
    });
  });

  /* ---------------- abstract toggles ---------------- */
  document.querySelectorAll('.toggle-abstract').forEach(function (btn) {
    btn.addEventListener('click', function () {
      var item = btn.closest('.pub-item');
      var open = item.classList.toggle('open');
      btn.textContent = open ? 'Hide abstract' : 'Read abstract';
    });
  });

  /* ---------------- theme toggle ---------------- */
  var themeToggle = document.getElementById('themeToggle');
  var root = document.documentElement;
  var stored = null;
  try { stored = localStorage.getItem('cello-theme'); } catch (e) { /* ignore */ }
  if (stored === 'light' || stored === 'dark') {
    root.setAttribute('data-theme', stored);
  }
  themeToggle.addEventListener('click', function () {
    var current = root.getAttribute('data-theme');
    var prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    var effectiveCurrent = current || (prefersDark ? 'dark' : 'light');
    var next = effectiveCurrent === 'dark' ? 'light' : 'dark';
    root.setAttribute('data-theme', next);
    try { localStorage.setItem('cello-theme', next); } catch (e) { /* ignore */ }
  });
})();
