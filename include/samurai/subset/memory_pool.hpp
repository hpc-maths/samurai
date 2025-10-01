// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <memory>
#include <cassert>
#include <vector>
#include <algorithm>
#include <ranges>

#include <fmt/ranges.h>

template<typename T, class DefaultAllocator = std::allocator<T>>
class MemoryPool
{
public:
	using Allocator       = typename std::allocator_traits<DefaultAllocator>::template rebind_alloc<T>;
    using AllocatorTraits = std::allocator_traits<Allocator>;
    using Element         = typename AllocatorTraits::value_type;
    using Pointer         = typename AllocatorTraits::pointer;
    using const_Pointer   = typename AllocatorTraits::const_pointer;
    using Reference       = Element&;
    using const_Reference = const Element&;
    using Distance        = typename AllocatorTraits::difference_type;
    using Size            = typename AllocatorTraits::size_type;
	using OffsetRange     = std::ranges::iota_view<Distance, Distance>;
	
	static_assert(std::is_move_constructible<Element>::value or std::is_trivially_copyable<Element>::value);
	
	struct Chunk
	{		
		Distance offset;
		Size     size;
		bool     isFree;
		
		friend bool operator<(const Chunk& lhs, const Chunk& rhs) { return lhs.offset < rhs.offset; }
	};

    using ChunkIterator = typename std::vector<Chunk>::iterator;
    
    MemoryPool(const MemoryPool&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;
    
	~MemoryPool();
    
    OffsetRange requestChunk(const Size chunkSize);
    
    void freeChunk(const OffsetRange offsets);
    
    Pointer       getPtr(const Distance offset)       { assert(Size(offset) < m_totalSize); return m_memory + offset; }
    const_Pointer getPtr(const Distance offset) const { assert(Size(offset) < m_totalSize); return m_memory + offset; }

    Reference       at(const Distance offset)       { return *getPtr(offset); }
    const_Reference at(const Distance offset) const { return *getPtr(offset); }
    
    Reference       operator[](const Distance offset)       { return at(offset); }
    const_Reference operator[](const Distance offset) const { return at(offset); }
    
    Size totalSize() const { return m_totalSize; }
    
    static MemoryPool& getInstance();
private:
	MemoryPool() {}

    Pointer            m_memory    = nullptr;
    Size               m_totalSize = 0;
    Allocator          m_allocator;
    std::vector<Chunk> m_chunks;
};

template<typename T, class DefaultAllocator>
MemoryPool<T,DefaultAllocator>& MemoryPool<T,DefaultAllocator>::getInstance()
{
	static MemoryPool instance; 
	
	return instance;
}

template<typename T, class DefaultAllocator>
MemoryPool<T,DefaultAllocator>::~MemoryPool()
{
	assert(std::all_of(std::begin(m_chunks), std::end(m_chunks), [](const Chunk& chunk) -> bool 
	{  
		return chunk.isFree;
	}));
	
	m_allocator.deallocate(m_memory, m_totalSize);
	
	m_memory    = nullptr;
	m_totalSize = 0;
}

template<typename T, class DefaultAllocator>
auto MemoryPool<T,DefaultAllocator>::requestChunk(const Size chunkSize) -> OffsetRange
{
	const ChunkIterator chunkIt = std::find_if(std::begin(m_chunks), std::end(m_chunks), [chunkSize](const Chunk& chunk) -> bool
	{
		return chunk.isFree and chunk.size >= chunkSize;
	});
	
	if (chunkIt != std::end(m_chunks))
	{		
		chunkIt->isFree = false;
		const OffsetRange ret(chunkIt->offset, chunkIt->offset + Distance(chunkSize));
		
		if (chunkIt->size > chunkSize)
		{
			// slipt the chunk
			const Distance newChunkOffset = chunkIt->offset + Distance(chunkSize);
			const Size     newChunkSize   = chunkIt->size - chunkSize;
			
			chunkIt->size = chunkSize;
			m_chunks.push_back({.offset = newChunkOffset, .size = newChunkSize, .isFree = true});
		}
		
		fmt::print("returning chunk {} of {}\n", fmt::join(ret, ", "), typeid(T).name());
		
		return ret;
	}
	else if (m_memory != nullptr)
	{
		const Size newTotalSize = std::max(m_totalSize + chunkSize, 2*m_totalSize);
		// we need to reallocate a larger buffer
		Pointer newMemory = m_allocator.allocate(newTotalSize);
		// we need to move all the data from our buffer to the new one
		auto usedChunks = m_chunks | std::views::filter([] (const Chunk& chunk) { return not chunk.isFree; });
		
		for (const Chunk& chunk : usedChunks)
		{
			Pointer beginChunk = m_memory   + chunk.offset;
			Pointer endChunk   = beginChunk + chunk.size;
			Pointer dstIt      = newMemory + chunk.offset;
			if constexpr (std::is_trivially_copyable<Element>::value)
			{
				std::copy(beginChunk, endChunk, dstIt);
				std::destroy(beginChunk, endChunk);
			}
			else
			{
				for (Pointer srcIt=beginChunk; srcIt!=endChunk; ++srcIt, ++dstIt)
				{
					std::construct_at(dstIt, std::move(*srcIt)); 
					std::destroy_at(srcIt);
				}
			}
		}
		// now we can swap the pointers and free newMemory
		std::swap(m_memory, newMemory);
		m_allocator.deallocate(newMemory, m_totalSize);
		
		// now we create a new chunk with the correct size
		
		m_chunks.push_back({.offset = Distance(m_totalSize), .size = chunkSize, .isFree = false});
		OffsetRange ret(m_chunks.back().offset, m_chunks.back().offset + Distance(m_chunks.back().size));
		
		if (newTotalSize != (m_totalSize + chunkSize))
		{
			m_chunks.push_back({.offset = Distance(m_totalSize + chunkSize), .size = Size(newTotalSize - m_totalSize - chunkSize), .isFree = true});
		}
		m_totalSize = newTotalSize;
		
		fmt::print("returning chunk {} of {}\n", fmt::join(ret, ", "), typeid(T).name());
		
		return ret; 
	}
	else
	{
		m_memory = m_allocator.allocate(chunkSize);
		m_totalSize = chunkSize;
		
		m_chunks.push_back({.offset = 0, .size = chunkSize, .isFree = false});
		
		fmt::print("returning chunk {} of {}\n", fmt::join(OffsetRange(m_chunks.back().offset, m_chunks.back().offset + Distance(m_chunks.back().size)), ", "), typeid(T).name());
		
		return OffsetRange(m_chunks.back().offset, m_chunks.back().offset + Distance(m_chunks.back().size));
	}
}

template<typename T, class DefaultAllocator>
auto MemoryPool<T,DefaultAllocator>::freeChunk(const OffsetRange offsets) -> void
{	
	const ChunkIterator chunkIt = std::find_if(std::begin(m_chunks), std::end(m_chunks), [&offsets](const Chunk& chunk) -> bool
	{
		return (not chunk.isFree) and chunk.offset == offsets[0] and chunk.size == offsets.size();
	});
	
	fmt::print("freeing {} of {}\n", fmt::join(offsets, ", "), typeid(T).name());
	
	assert(chunkIt != std::end(m_chunks));
	
	if (chunkIt != std::end(m_chunks))
	{
		chunkIt->isFree = true;
		
		std::destroy(m_memory + chunkIt->offset,  m_memory + chunkIt->offset + chunkIt->size);
		
		// Coalescing
		std::sort(std::begin(m_chunks), std::end(m_chunks));
		
		for (ChunkIterator it=std::begin(m_chunks); it!=std::end(m_chunks); ++it)
		{
			if (it->isFree)
			{
				for (auto nextIt=std::next(it); nextIt!=std::end(m_chunks) and nextIt->isFree and it->offset + Distance(it->size) == nextIt->offset; ++nextIt)
				{
					it->size += nextIt->size;
					nextIt->size = 0; // mark for removal with std::remove_if
				}
			}
		}
		
		const ChunkIterator newEnd = std::remove_if(std::begin(m_chunks), std::end(m_chunks), [](const Chunk& chunk) -> bool
		{
			return chunk.size == 0;
		});
		m_chunks.erase(newEnd, std::end(m_chunks));
	}
}
