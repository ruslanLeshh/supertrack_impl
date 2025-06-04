from gym.envs.registration import register
# registering allows gym.make()
register(
    id='SuperTrack-v0',
    entry_point='supertrack.envs.supertrack_env:SuperTrackEnv'
)