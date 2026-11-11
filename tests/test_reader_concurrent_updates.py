"""
Tests for concurrent update protection.

Enabled with make_reader(concurrent_updates=True); see
ConcurrentUpdateError and Storage.update_feed_guarded().

The storage-level tests use one database file and real SQLite connections
from multiple threads; the multiprocess gate test at the bottom uses
separate processes to prove the protection works across processes too.

"""

import multiprocessing
import sqlite3
import warnings

import pytest

from reader import ConcurrentUpdateError
from reader import FeedNotFoundError
from reader import ParseError
from reader import StorageError
from reader import UpdateError
from reader import UpdateHookError
from reader._storage import Storage
from reader._types import EntryUpdateIntent
from reader._types import FeedUpdateIntent
from reader._types import FeedToUpdate
from utils import utc_datetime as datetime


# ---------------------------------------------------------------------------
# storage-level generation bookkeeping
# ---------------------------------------------------------------------------


def test_v44_database_migrates_and_keeps_working(db_path):
    """Old databases auto-upgrade: the column and counter appear,
    existing feeds get a generation, and new feeds get fresh ones."""
    from reader._storage._schema import HeavyMigration

    # create a current-schema database with a feed, then roll it back to v44
    storage = Storage(db_path)
    storage.add_feed('old', datetime(2010, 1, 1))
    HeavyMigration.set_version(storage.get_db(), 44)
    storage.get_db().execute("DROP TABLE meta;")
    storage.get_db().execute(
        "ALTER TABLE feeds DROP COLUMN update_generation;"
    )
    storage.close()

    # opening runs the 44 -> 45 migration automatically
    migrated = Storage(db_path)
    assert HeavyMigration.get_version(migrated.get_db()) == 45

    (old_gen,) = migrated.get_db().execute(
        "SELECT update_generation FROM feeds WHERE url = 'old';"
    ).fetchone()
    assert old_gen == 1  # column default for pre-existing rows

    (next_generation,) = migrated.get_db().execute(
        "SELECT value FROM meta WHERE key = 'next_update_generation';"
    ).fetchone()
    assert next_generation == 2

    # the migrated storage works end to end, and never reuses generations
    migrated.add_feed('new', datetime(2010, 1, 2))
    (new_gen,) = migrated.get_db().execute(
        "SELECT update_generation FROM feeds WHERE url = 'new';"
    ).fetchone()
    assert new_gen != old_gen

    # guarded commits work on the migrated database
    migrated.update_feed_guarded(
        _feed_intent('old'), [], old_gen
    )
    migrated.close()


def test_generations_are_unique_and_monotonic(storage):
    storage.add_feed('one', datetime(2010, 1, 1))
    storage.add_feed('two', datetime(2010, 1, 1))

    gens = dict(
        storage.get_db().execute("SELECT url, update_generation FROM feeds;")
    )
    assert gens['one'] != gens['two']

    storage.delete_feed('one')
    # re-adding the same URL must not reuse the old generation
    storage.add_feed('one', datetime(2010, 1, 2))
    (new_gen,) = storage.get_db().execute(
        "SELECT update_generation FROM feeds WHERE url = 'one';"
    ).fetchone()
    assert new_gen != gens['one']

    storage.change_feed_url('two', 'three')
    (moved_gen,) = storage.get_db().execute(
        "SELECT update_generation FROM feeds WHERE url = 'three';"
    ).fetchone()
    assert moved_gen != gens['two']
    assert len({gens['one'], gens['two'], new_gen, moved_gen}) == 4


def test_change_feed_url_bumps_generation(storage):
    storage.add_feed('one', datetime(2010, 1, 1))
    (before,) = storage.get_db().execute(
        "SELECT update_generation FROM feeds WHERE url = 'one';"
    ).fetchone()
    storage.change_feed_url('one', 'two')
    (after,) = storage.get_db().execute(
        "SELECT update_generation FROM feeds WHERE url = 'two';"
    ).fetchone()
    assert after > before


# ---------------------------------------------------------------------------
# storage-level CAS commit
# ---------------------------------------------------------------------------


def _feed_intent(url='feed', value=None):
    return FeedUpdateIntent(
        url,
        datetime(2010, 1, 2),
        datetime(2010, 1, 3),
        value if value is not None
        else FeedToUpdate(
            feed=_feed_data(url),
            last_updated=datetime(2010, 1, 2),
        ),
    )


def _feed_data(url):
    from reader._types import FeedData

    return FeedData(url, title='title')


def _entry_intent(feed='feed', id='entry', title='title'):
    from reader._types import EntryData

    return EntryUpdateIntent(
        EntryData(feed, id, title=title),
        datetime(2010, 1, 2),
        datetime(2010, 1, 1),
        datetime(2010, 1, 1),
        datetime(2010, 1, 1),
    )


def _generation(storage, url='feed'):
    (gen,) = storage.get_db().execute(
        "SELECT update_generation FROM feeds WHERE url = ?;", (url,)
    ).fetchone()
    return gen


def test_update_feed_guarded_commits_when_generation_matches(storage):
    storage.add_feed('feed', datetime(2010, 1, 1))
    gen = _generation(storage)

    storage.update_feed_guarded(
        _feed_intent(), [_entry_intent()], gen
    )

    assert storage.get_entry_recent_sort(('feed', 'entry')) == datetime(2010, 1, 1)
    # generation unchanged by a successful commit
    assert _generation(storage) == gen


def test_update_feed_guarded_raises_on_stale_generation(storage):
    storage.add_feed('feed', datetime(2010, 1, 1))
    gen = _generation(storage)

    # the feed "changed" (e.g. another update won) before the guarded commit
    db = storage.get_db()
    with db:
        db.execute(
            "UPDATE feeds SET update_generation = ? WHERE url = ?;",
            (gen + 100, 'feed'),
        )

    with pytest.raises(ConcurrentUpdateError) as excinfo:
        storage.update_feed_guarded(
            _feed_intent(), [_entry_intent()], gen
        )
    assert excinfo.value.url == 'feed'
    assert isinstance(excinfo.value, UpdateError)

    # nothing was written: the entry must not exist
    assert list(storage.get_entries_for_update([('feed', 'entry')])) == [None]
    rows = storage.get_db().execute(
        "SELECT id FROM entries WHERE feed = 'feed';"
    ).fetchall()
    assert rows == []


def test_update_feed_guarded_raises_feed_not_found_when_deleted(storage):
    storage.add_feed('feed', datetime(2010, 1, 1))
    gen = _generation(storage)
    storage.delete_feed('feed')

    with pytest.raises(FeedNotFoundError):
        storage.update_feed_guarded(_feed_intent(), [], gen)


class _FailingStorage(Storage):
    """Fail the guarded commit after the feed CAS succeeded, while entries
    are being written (Connection.execute is read-only on Python 3.14,
    so the failing statement is injected via the entry-write hook)."""

    def _insert_entry_dict(self, db, intent):
        raise sqlite3.OperationalError("boom")


def test_update_feed_guarded_rolls_back_on_write_failure():
    storage = _FailingStorage(':memory:')
    storage.add_feed('feed', datetime(2010, 1, 1))
    gen = _generation(storage)

    with pytest.raises(StorageError):
        storage.update_feed_guarded(
            _feed_intent(), [_entry_intent()], gen
        )

    # atomic rollback: neither the entries nor the feed CAS update persist
    assert storage.get_db().execute(
        "SELECT count(*) FROM entries WHERE feed = 'feed';"
    ).fetchone()[0] == 0
    row = storage.get_db().execute(
        "SELECT last_retrieved, last_exception FROM feeds WHERE url = 'feed';"
    ).fetchone()
    assert row == (None, None)
    storage.close()


# ---------------------------------------------------------------------------
# reader-level behavior (fakeparser)
# ---------------------------------------------------------------------------


def _guarded_reader(make_reader, db_path, parser):
    reader = make_reader(db_path, feed_root='', concurrent_updates=True)
    reader._parser = parser
    return reader


def _install_race_winner(monkeypatch, db_path, losing_url):
    """Make any update of ``losing_url`` lose: a separate connection
    commits a generation bump while the update is in store_feed(),
    i.e. after the update read its (now stale) generation."""
    from reader._update import Pipeline

    original_store_feed = Pipeline.store_feed

    def store_feed(self, feed, entries, generation=None):
        if feed.url == losing_url:
            with Storage(db_path) as winner:
                db = winner.get_db()
                (current,) = db.execute(
                    "SELECT update_generation FROM feeds WHERE url = ?;",
                    (losing_url,),
                ).fetchone()
                with db:
                    db.execute(
                        "UPDATE feeds SET update_generation = ? WHERE url = ?;",
                        (current + 1, losing_url),
                    )
        return original_store_feed(self, feed, entries, generation)

    monkeypatch.setattr(Pipeline, 'store_feed', store_feed)


def test_flag_defaults_off(make_reader, db_path):
    reader = make_reader(db_path)
    assert reader._concurrent_updates is False


def test_flag_enabled(make_reader, db_path):
    reader = make_reader(db_path, concurrent_updates=True)
    assert reader._concurrent_updates is True


def test_guarded_update_normal_path(make_reader, db_path, parser):
    reader = _guarded_reader(make_reader, db_path, parser)
    feed = parser.feed(1)
    entry = parser.entry(1, 1, title='one')
    reader.add_feed(feed)

    result = reader.update_feed(feed)
    assert result.new == 1
    assert reader.get_entry(entry).title == 'one'


@pytest.mark.parametrize('outcome', ['ok', 'not_modified', 'parse_error'])
def test_late_result_loses_in_all_outcomes(
    make_reader, db_path, parser, monkeypatch, outcome
):
    """200 (ok), 304 (not modified) and retrieval failure all get discarded
    if a newer update committed first."""

    from reader._update import Pipeline

    reader = _guarded_reader(make_reader, db_path, parser)
    feed = parser.feed(1, title='first')
    entry = parser.entry(1, 1, title='first-entry')
    reader.add_feed(feed)
    reader.update_feed(feed)

    original_store_feed = Pipeline.store_feed

    def store_feed(*args, **kwargs):
        # Just before this (late) result commits, a different connection
        # wins the race by bumping the generation, then commits.
        with Storage(db_path) as winner:
            db = winner.get_db()
            (gen,) = db.execute(
                "SELECT update_generation FROM feeds WHERE url = ?;",
                (feed.url,),
            ).fetchone()
            with db:
                db.execute(
                    "UPDATE feeds SET update_generation = ? WHERE url = ?;",
                    (gen + 1, feed.url),
                )
        return original_store_feed(*args, **kwargs)

    monkeypatch.setattr(Pipeline, 'store_feed', store_feed)

    if outcome == 'not_modified':
        parser.not_modified(lambda url: True)
    elif outcome == 'parse_error':
        parser.raise_exc(ParseError(feed.url, message='nope'))

    with pytest.raises(ConcurrentUpdateError):
        reader.update_feed(feed)

    # the previously committed content is untouched
    assert reader.get_entry(entry).title == 'first-entry'


def test_single_update_raises_recognizable_error(
    make_reader, db_path, parser, monkeypatch
):
    reader = _guarded_reader(make_reader, db_path, parser)
    feed = parser.feed(1)
    reader.add_feed(feed)
    reader.update_feed(feed)

    _install_race_winner(monkeypatch, db_path, feed.url)

    # update_feed() raises the recognizable UpdateError
    with pytest.raises(ConcurrentUpdateError):
        reader.update_feed(feed)


def test_iter_yields_conflict_and_continues(make_reader, db_path, parser, monkeypatch):
    reader = _guarded_reader(make_reader, db_path, parser)
    losing = parser.feed(1)
    winning = parser.feed(2)
    reader.add_feed(losing)
    reader.add_feed(winning)
    reader.update_feed(losing)
    reader.update_feed(winning)

    _install_race_winner(monkeypatch, db_path, losing.url)

    results = list(reader.update_feeds_iter(updates_enabled=None, scheduled=False))
    by_url = dict(results)

    assert isinstance(by_url[losing.url], ConcurrentUpdateError)
    # feed 2 still updated successfully
    assert not isinstance(by_url[winning.url], Exception)


def test_batch_warns_and_continues(make_reader, db_path, parser, monkeypatch):
    reader = _guarded_reader(make_reader, db_path, parser)
    losing = parser.feed(1)
    winning = parser.feed(2)
    parser.entry(1, 1, title='losing-entry')
    parser.entry(2, 1, title='winning-entry')
    reader.add_feed(losing)
    reader.add_feed(winning)
    reader.update_feed(losing)
    reader.update_feed(winning)

    _install_race_winner(monkeypatch, db_path, losing.url)

    # change winning feed content so it has something to commit
    parser.entry(2, 1, title='winning-entry-2')

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        reader.update_feeds(updates_enabled=None, scheduled=False)

    conflict_warnings = [
        w for w in caught if issubclass(w.category, ConcurrentUpdateError)
    ]
    assert len(conflict_warnings) == 1
    assert conflict_warnings[0].message.url == losing.url

    # the winning feed was still updated
    assert reader.get_entry((winning.url, '2, 1')).title == 'winning-entry-2'


def test_no_automatic_retry(make_reader, db_path, parser, monkeypatch):
    """A conflict must not trigger another retrieve/update attempt."""
    reader = _guarded_reader(make_reader, db_path, parser)
    feed = parser.feed(1)
    reader.add_feed(feed)
    reader.update_feed(feed)

    _install_race_winner(monkeypatch, db_path, feed.url)

    retrieve_calls = []
    real_retrieve = reader._parser.retrieve

    def counting_retrieve(url, caching_info):
        retrieve_calls.append(url)
        return real_retrieve(url, caching_info)

    reader._parser.retrieve = counting_retrieve

    with pytest.raises(ConcurrentUpdateError):
        reader.update_feed(feed)

    # exactly one retrieve: no automatic re-pull after the conflict
    assert retrieve_calls == [feed.url]


def test_delete_and_readd_isolates_inflight(
    make_reader, db_path, parser, monkeypatch
):
    reader = _guarded_reader(make_reader, db_path, parser)
    feed = parser.feed(1, title='old')
    parser.entry(1, 1, title='old-entry')
    reader.add_feed(feed)
    reader.update_feed(feed)

    from reader._update import Pipeline

    original_store_feed = Pipeline.store_feed
    raced = False

    def store_feed(*args, **kwargs):
        nonlocal raced
        # delete and re-add the same URL while this update is in flight;
        # the re-added feed gets a brand-new, never-reused generation.
        # Only the outer (losing) call injects the race; the inner
        # update_feed() below must run normally.
        if not raced:
            raced = True
            reader.delete_feed(feed)
            fresh = _guarded_reader(make_reader, db_path, parser.copy())
            fresh.add_feed(feed)
            fresh.update_feed(feed)
            fresh.close()
        return original_store_feed(*args, **kwargs)

    monkeypatch.setattr(Pipeline, 'store_feed', store_feed)

    with pytest.raises(ConcurrentUpdateError):
        reader.update_feed(feed)

    # the freshly committed feed must be intact, not overwritten by the
    # in-flight result of the deleted generation
    fresh_reader = _guarded_reader(make_reader, db_path, parser)
    assert fresh_reader.get_feed(feed).title == 'old'
    fresh_reader.close()


def test_change_url_away_and_back_isolates_inflight(
    make_reader, db_path, parser, monkeypatch
):
    reader = _guarded_reader(make_reader, db_path, parser)
    feed = parser.feed(1, title='orig')
    reader.add_feed(feed)
    reader.update_feed(feed)

    from reader._update import Pipeline

    original_store_feed = Pipeline.store_feed
    raced = False

    def store_feed(*args, **kwargs):
        nonlocal raced
        # change away and back while this update is in flight;
        # each change bumps the generation, so the stale result is rejected.
        # Guard against re-entry in case the change path itself stores.
        if not raced:
            raced = True
            reader.change_feed_url(feed, '2')
            reader.change_feed_url('2', feed)
        return original_store_feed(*args, **kwargs)

    monkeypatch.setattr(Pipeline, 'store_feed', store_feed)

    with pytest.raises(ConcurrentUpdateError):
        reader.update_feed(feed)


def test_read_important_tags_preserved_on_conflict(
    make_reader, db_path, parser, monkeypatch
):
    """A discarded update must not touch user data on the winning state."""
    reader = _guarded_reader(make_reader, db_path, parser)
    feed = parser.feed(1)
    entry = parser.entry(1, 1, title='kept')
    reader.add_feed(feed)
    reader.update_feed(feed)

    reader.mark_entry_as_read(entry)
    reader.mark_entry_as_important(entry)
    reader.set_tag((feed.url, entry.id), 'tag', 'value')

    _install_race_winner(monkeypatch, db_path, feed.url)

    # a conflicting result carrying a changed title gets discarded
    parser.entry(1, 1, title='should-not-win')
    with pytest.raises(ConcurrentUpdateError):
        reader.update_feed(feed)

    stored = reader.get_entry(entry)
    assert stored.title == 'kept'
    assert stored.read is True
    assert stored.important is True
    assert reader.get_tag((feed.url, entry.id), 'tag') == 'value'


def test_after_hooks_run_only_after_commit(
    make_reader, db_path, parser, monkeypatch
):
    reader = _guarded_reader(make_reader, db_path, parser)
    feed = parser.feed(1)
    reader.add_feed(feed)
    reader.update_feed(feed)

    events = []
    reader.after_feed_update_hooks.append(
        lambda r, url: events.append(('after_feed', url))
    )
    reader.before_feed_update_hooks.append(
        lambda r, url: events.append(('before_feed', url))
    )

    _install_race_winner(monkeypatch, db_path, feed.url)

    with pytest.raises(ConcurrentUpdateError):
        reader.update_feed(feed)

    # before hook ran, after hook did not (no successful commit)
    assert ('before_feed', feed.url) in events
    assert not any(kind == 'after_feed' for kind, _ in events)


def test_after_hook_exception_does_not_rollback(make_reader, db_path, parser):
    reader = _guarded_reader(make_reader, db_path, parser)
    feed = parser.feed(1)
    entry = parser.entry(1, 1, title='hook-era')
    reader.add_feed(feed)

    def boom(r, url):
        raise RuntimeError("hook exploded")

    reader.after_feed_update_hooks.append(boom)

    # the hook failure is surfaced, but the commit stays in place
    with pytest.raises(UpdateHookError):
        reader.update_feed(feed)

    assert reader.get_entry(entry).title == 'hook-era'


def test_retry_after_conflict_succeeds(
    make_reader, db_path, parser, monkeypatch
):
    reader = _guarded_reader(make_reader, db_path, parser)
    feed = parser.feed(1, title='v1')
    reader.add_feed(feed)
    reader.update_feed(feed)

    # the first attempt loses a race; the monkeypatch is removed afterwards,
    # so the next update reads the current generation and commits.
    _install_race_winner(monkeypatch, db_path, feed.url)

    parser.feed(1, title='v2')
    with pytest.raises(ConcurrentUpdateError):
        reader.update_feed(feed)

    monkeypatch.undo()

    # a fresh update reads the current generation and commits normally
    result = reader.update_feed(feed)
    assert result is not None
    assert reader.get_feed(feed).title == 'v2'


def test_default_reader_keeps_last_writer_wins(make_reader, db_path, parser):
    """With the flag off, behavior is unchanged: no ConcurrentUpdateError."""
    reader = make_reader(db_path, feed_root='')
    reader._parser = parser
    feed = parser.feed(1)
    reader.add_feed(feed)
    reader.update_feed(feed)

    # a generation mismatch is ignored entirely on the legacy path
    db = reader._storage.get_db()
    gen = _generation(reader._storage, feed.url)
    with db:
        db.execute(
            "UPDATE feeds SET update_generation = ? WHERE url = ?;",
            (gen + 1, feed.url),
        )

    # no exception despite the stale generation (legacy path)
    reader.update_feed(feed)


# ---------------------------------------------------------------------------
# multiprocess gate
# ---------------------------------------------------------------------------


_TESTS_DIR = __file__.rsplit('/', 1)[0]


@pytest.mark.slow
@pytest.mark.parametrize('mode', ['ok', 'not_modified', 'parse_error'])
def test_multiprocess_concurrent_gate(make_reader, db_path, parser, mode):
    """Two processes fetch the same feed; whichever commits first wins,
    and the late process gets ConcurrentUpdateError without writing anything.
    Works for 200, 304 and retrieval-failure results."""

    ctx = multiprocessing.get_context('spawn')

    reader = make_reader(db_path, feed_root='', concurrent_updates=True)
    reader._parser = parser
    feed = parser.feed(1, title='main')
    entry = parser.entry(1, 1, title='main-entry')
    reader.add_feed(feed)
    reader.update_feed(feed)

    ready = ctx.Event()
    proceed = ctx.Event()
    queue = ctx.Queue()

    proc = ctx.Process(
        target=_mp_worker,
        args=(db_path, feed.url, ready, proceed, mode, queue),
    )
    proc.start()
    try:
        assert ready.wait(timeout=30)

        # the child is blocked in network retrieve; meanwhile the main
        # process commits a newer generation (no write lock is held while
        # waiting, so this must not block)
        with Storage(db_path) as winner:
            db = winner.get_db()
            (gen,) = db.execute(
                "SELECT update_generation FROM feeds WHERE url = ?;",
                (feed.url,),
            ).fetchone()
            with db:
                db.execute(
                    "UPDATE feeds SET update_generation = ? WHERE url = ?;",
                    (gen + 1, feed.url),
                )

        proceed.set()
        proc.join(timeout=30)
        assert not proc.is_alive()
        outcome = queue.get(timeout=10)
    finally:
        if proc.is_alive():
            proc.terminate()
            proc.join()

    assert outcome == 'conflict', outcome

    # main-process data untouched by the discarded child result
    assert reader.get_entry(entry).title == 'main-entry'


def _mp_worker(db_path, feed_url, ready, proceed, mode, queue):  # pragma: no cover
    import sys

    sys.path.insert(0, _TESTS_DIR)
    from fakeparser import Parser as _Parser

    from reader import make_reader

    parser = _Parser()
    parser.feed(1, title='worker')
    parser.entry(1, 1, title='worker-entry')

    if mode == 'not_modified':
        parser.not_modified(lambda url: True)
    elif mode == 'parse_error':
        from reader import ParseError as _ParseError

        parser.raise_exc(_ParseError(feed_url, message='down'))

    reader = make_reader(db_path, feed_root='', concurrent_updates=True)
    reader._parser = parser

    original = reader._parser.retrieve

    def blocked_retrieve(url, caching_info):
        ready.set()
        proceed.wait(timeout=30)
        return original(url, caching_info)

    reader._parser.retrieve = blocked_retrieve

    try:
        reader.update_feed(feed_url)
        queue.put('ok')
    except ConcurrentUpdateError:
        queue.put('conflict')
    except Exception as e:  # noqa: BLE001
        queue.put(f'{type(e).__name__}: {e}')
    finally:
        reader.close()
